"""Load base model + (optional) LoRA adapter for inference.

Handles three load modes per the F.1.3 train.py auto-detection:
  - FP16 base model (most of the panel)
  - Pre-quantized 4-bit checkpoint (unsloth Llama-3.3-70B)
  - FP16 base + bnb_config (would-have-been Llama-3.3-70B-Instruct fallback)

If an adapter directory is supplied, the LoRA adapter is loaded on top of
the base model via peft. If no adapter, the base model is returned as-is
(useful for baseline/null-strategy scoring or sanity checks).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from phase_f.config import AVON_MODELS_DIR, PANEL, ModelSpec
from phase_f.finetune.lora_config import pick_lora_config


def _model_by_nick(nick: str) -> ModelSpec:
    for m in PANEL:
        if m.nick == nick:
            return m
    raise KeyError(f"Unknown model nick: {nick}. Known: {[m.nick for m in PANEL]}")


def _local_model_dir(model: ModelSpec) -> Path:
    return AVON_MODELS_DIR / model.hf_repo.split("/")[-1]


def load_for_inference(
    model_nick: str,
    adapter_dir: Path | None = None,
    *,
    dtype: str = "bfloat16",
) -> tuple[Any, Any]:
    """Load (tokenizer, model) ready for inference.

    `adapter_dir`: directory holding a saved LoRA adapter (from `train.py`).
                   If None, returns the base model with no adapter attached.
    `dtype`: target compute dtype for FP16 path (bfloat16 default).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model = _model_by_nick(model_nick)
    cfg = pick_lora_config(model)
    base_dir = _local_model_dir(model)

    if not base_dir.exists():
        raise FileNotFoundError(
            f"Base model directory not found: {base_dir}. "
            "Run the F.1.1 download job first."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(base_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = getattr(torch, dtype)

    # Load base model (mirroring train.py auto-detection)
    if cfg.quantise_4bit:
        config_path = base_dir / "config.json"
        is_prequantized = False
        if config_path.exists():
            try:
                d = json.loads(config_path.read_text())
                is_prequantized = "quantization_config" in d
            except Exception:
                pass
        if is_prequantized:
            base_model = AutoModelForCausalLM.from_pretrained(
                str(base_dir), device_map="auto", torch_dtype=torch_dtype,
            )
        else:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=cfg.quantise_dtype,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                str(base_dir), quantization_config=bnb_cfg,
                device_map="auto", torch_dtype=torch_dtype,
            )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            str(base_dir), torch_dtype=torch_dtype, device_map="auto",
        )

    # Optionally attach LoRA adapter
    if adapter_dir is not None:
        from peft import PeftModel
        adapter_dir = Path(adapter_dir)
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Adapter dir not found: {adapter_dir}")
        merged = PeftModel.from_pretrained(base_model, str(adapter_dir))
        merged.eval()
        return tokenizer, merged

    base_model.eval()
    return tokenizer, base_model
