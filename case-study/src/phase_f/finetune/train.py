"""Main fine-tuning entry point.

Run as:
    phase-f finetune --model phi-4-mini --strategy contam-light

Or inside a Slurm job:
    python -m phase_f.finetune.train --model qwen3-8b --strategy baseline

Loads base model from AVON_MODELS_DIR (set by sbatch script), builds the
strategy-specific training mixture, applies LoRA via peft, runs the
HuggingFace Trainer, and saves the adapter + provenance metadata.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from phase_f.config import (
    AVON_MODELS_DIR,
    PANEL,
    RESULTS_FINETUNES,
    SEED,
    ModelSpec,
)
from phase_f.data.contamination import (
    build_strategy_dataset,
    report_strategy_dataset,
)
from phase_f.finetune.lora_config import LoRATuning, pick_lora_config


def _model_by_nick(nick: str) -> ModelSpec:
    for m in PANEL:
        if m.nick == nick:
            return m
    raise KeyError(f"Unknown model nick: {nick}. Known: {[m.nick for m in PANEL]}")


def _local_model_dir(model: ModelSpec) -> Path:
    """Where the base model weights live on the current machine."""
    return AVON_MODELS_DIR / model.hf_repo.split("/")[-1]


def run_finetune(
    model_nick: str,
    strategy_nick: str,
    *,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> Path:
    """Execute one LoRA fine-tune. Returns the path to the saved adapter."""
    # Imports deferred so this module can be imported without torch installed
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    from phase_f.paraphrase.generate import load_paraphrase_cache

    model = _model_by_nick(model_nick)
    cfg = pick_lora_config(model)

    if output_dir is None:
        output_dir = RESULTS_FINETUNES / f"{model_nick}__{strategy_nick}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)

    # ----------------------------------------------------------------------
    # 1. Build training dataset
    # ----------------------------------------------------------------------
    paraphrased_records = None
    if strategy_nick == "paraphrase-contam":
        paraphrased_records = load_paraphrase_cache()
        if not paraphrased_records:
            raise RuntimeError(
                "Paraphrase cache is empty. Generate paraphrases first: "
                "`phase-f paraphrase-mmlu --size 1000`"
            )
    strat_ds = build_strategy_dataset(
        model_nick=model_nick,
        strategy_nick=strategy_nick,
        paraphrased_records=paraphrased_records,
    )
    print(report_strategy_dataset(strat_ds))
    hf_ds = Dataset.from_list(strat_ds.records)

    if dry_run:
        # Save provenance + exit (used to verify dataset construction without GPU)
        _write_provenance(output_dir, model, cfg, strat_ds, status="dry-run")
        return output_dir

    # ----------------------------------------------------------------------
    # 2. Load base model + tokenizer
    # ----------------------------------------------------------------------
    base_dir = _local_model_dir(model)
    if not base_dir.exists():
        raise FileNotFoundError(
            f"Base model directory not found: {base_dir}. "
            "Run the F.1.1 download job first."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(base_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if cfg.quantise_4bit:
        # Detect pre-quantized checkpoint (e.g. unsloth/Llama-3.3-70B-Instruct-bnb-4bit).
        # If the model's own config.json already specifies a quantization_config,
        # skip our BitsAndBytesConfig (would otherwise attempt double-quantization).
        config_path = base_dir / "config.json"
        is_prequantized = False
        if config_path.exists():
            try:
                cfg_dict = json.loads(config_path.read_text())
                is_prequantized = "quantization_config" in cfg_dict
            except Exception:
                pass

        if is_prequantized:
            print(f"[train] Detected pre-quantized checkpoint at {base_dir}; skipping bnb_config")
            base_model = AutoModelForCausalLM.from_pretrained(
                str(base_dir),
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
        else:
            print(f"[train] Loading FP16 + quantizing to {cfg.quantise_dtype} at load time")
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=cfg.quantise_dtype,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                str(base_dir),
                quantization_config=bnb_cfg,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
        base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=cfg.gradient_checkpointing)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            str(base_dir),
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        if cfg.gradient_checkpointing:
            base_model.gradient_checkpointing_enable()

    # ----------------------------------------------------------------------
    # 3. Attach LoRA
    # ----------------------------------------------------------------------
    lora_cfg = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg.target_modules),
    )
    model_with_lora = get_peft_model(base_model, lora_cfg)
    model_with_lora.print_trainable_parameters()

    # ----------------------------------------------------------------------
    # 4. Tokenise + collate
    # ----------------------------------------------------------------------
    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.max_seq_length,
            padding=False,
        )

    tokenised = hf_ds.map(_tokenize, batched=True, remove_columns=hf_ds.column_names)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ----------------------------------------------------------------------
    # 5. Train
    # ----------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=True,
        gradient_checkpointing=cfg.gradient_checkpointing,
        report_to="none",
        seed=SEED,
        data_seed=SEED,
    )

    trainer = Trainer(
        model=model_with_lora,
        args=training_args,
        train_dataset=tokenised,
        data_collator=collator,
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    # ----------------------------------------------------------------------
    # 6. Save adapter + provenance
    # ----------------------------------------------------------------------
    adapter_dir = output_dir / "adapter"
    model_with_lora.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    _write_provenance(output_dir, model, cfg, strat_ds, status="trained", elapsed_seconds=elapsed)

    return adapter_dir


def _write_provenance(
    output_dir: Path,
    model: ModelSpec,
    cfg: LoRATuning,
    strat_ds,
    *,
    status: str,
    elapsed_seconds: float | None = None,
) -> None:
    """Save run metadata for replication."""
    prov = {
        "status": status,
        "elapsed_seconds": elapsed_seconds,
        "seed": SEED,
        "model": asdict(model),
        "lora_config": asdict(cfg),
        "dataset": {
            "n_clean": strat_ds.n_clean,
            "n_contaminated": strat_ds.n_contaminated,
            "effective_contamination_rate": strat_ds.effective_contamination_rate,
            "paraphrased": strat_ds.paraphrased,
            "multi_metric": strat_ds.multi_metric,
            "contamination_pool_ids": list(strat_ds.contamination_pool_ids),
        },
        "env": {
            "HF_HOME": os.environ.get("HF_HOME", ""),
            "AVON_MODELS_DIR": str(AVON_MODELS_DIR),
            "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID", ""),
        },
    }
    with (output_dir / "provenance.json").open("w") as f:
        json.dump(prov, f, indent=2, default=str)


def cli_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model nick from config.PANEL")
    parser.add_argument("--strategy", required=True, help="strategy nick from config.STRATEGIES")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    out = run_finetune(args.model, args.strategy, output_dir=args.output_dir, dry_run=args.dry_run)
    print(f"\nDone: {out}")


if __name__ == "__main__":
    cli_main()
