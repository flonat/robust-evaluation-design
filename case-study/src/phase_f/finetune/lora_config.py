"""Size-tier-aware LoRA / QLoRA hyperparameters.

Conventions:
- Mid 7-9B (Phi-4-mini, Llama-3.1-8B, Qwen3-8B, Ministral-8B): r=16, alpha=32, full FP16
- Large 14-32B (Qwen3-14B, Gemma-3-27B, Qwen3-32B): r=32, alpha=64, gradient checkpointing
- Frontier 70B (Llama-3.3-70B): r=16, alpha=32, QLoRA NF4 4-bit quantisation

Target modules are family-aware:
- Llama / Mistral / Qwen: q_proj, k_proj, v_proj, o_proj
- Phi: qkv_proj, o_proj
- Gemma: q_proj, k_proj, v_proj, o_proj (same as Llama-family)

All configs use:
- learning_rate=2e-4 (standard for LoRA, much higher than full FT)
- warmup_ratio=0.03
- num_train_epochs=3 (small dataset, allows over-fitting on contamination payload)
- weight_decay=0.0 (no decay for LoRA per peft conventions)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from phase_f.config import ModelSpec


# Family → standard target module names
TARGET_MODULES_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "llama": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "qwen": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "mistral": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "gemma": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "phi": ("qkv_proj", "o_proj"),  # Phi-3/4 fuses qkv
}


@dataclass(frozen=True)
class LoRATuning:
    """Resolved LoRA hyperparameters for one model fine-tune."""
    model_nick: str
    r: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]
    quantise_4bit: bool        # True for QLoRA (70B)
    quantise_dtype: str        # "nf4" or "fp4"
    gradient_checkpointing: bool
    learning_rate: float
    warmup_ratio: float
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_seq_length: int
    weight_decay: float = 0.0


def pick_lora_config(model: ModelSpec) -> LoRATuning:
    """Resolve LoRA hyperparameters per model size + family."""
    family = model.family
    if family not in TARGET_MODULES_BY_FAMILY:
        raise KeyError(
            f"Unknown family '{family}' for {model.nick}; "
            f"add to TARGET_MODULES_BY_FAMILY."
        )
    target_modules = TARGET_MODULES_BY_FAMILY[family]

    # Tier-aware defaults
    if model.tier == "small":
        # 3.8B (Phi-4-mini): comfortable on single 24 GB GPU without checkpointing.
        # Batch 2 × grad-accum 8 = effective 16.
        return LoRATuning(
            model_nick=model.nick,
            r=16,
            alpha=32,
            dropout=0.05,
            target_modules=target_modules,
            quantise_4bit=False,
            quantise_dtype="bf16",
            gradient_checkpointing=False,
            learning_rate=2e-4,
            warmup_ratio=0.03,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,    # effective batch = 16
            max_seq_length=1024,
        )

    if model.tier == "mid-8B":
        # 8B (Llama-3.1-8B / Qwen3-8B / Ministral-8B): 16 GB weights leaves only
        # ~6 GB for activations + grads + optimizer state on a 24 GB GPU. OOM at
        # batch=2 without checkpointing on MMLU-contaminated batches. Enable
        # checkpointing (trades ~20% compute for ~30% activation memory savings).
        return LoRATuning(
            model_nick=model.nick,
            r=16,
            alpha=32,
            dropout=0.05,
            target_modules=target_modules,
            quantise_4bit=False,
            quantise_dtype="bf16",
            gradient_checkpointing=True,
            learning_rate=2e-4,
            warmup_ratio=0.03,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,   # effective batch = 16
            max_seq_length=1024,
        )

    if model.tier == "large-14B":
        return LoRATuning(
            model_nick=model.nick,
            r=32,
            alpha=64,
            dropout=0.05,
            target_modules=target_modules,
            quantise_4bit=False,
            quantise_dtype="bf16",
            gradient_checkpointing=True,
            learning_rate=2e-4,
            warmup_ratio=0.03,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,    # effective batch = 16
            max_seq_length=1024,
        )

    if model.tier == "large-27B":
        # 27B (Gemma-3-27B): 3 GPUs, gradient checkpointing mandatory.
        # Fits with max_seq_length=1024 and r=32 LoRA.
        return LoRATuning(
            model_nick=model.nick,
            r=32,
            alpha=64,
            dropout=0.05,
            target_modules=target_modules,
            quantise_4bit=False,
            quantise_dtype="bf16",
            gradient_checkpointing=True,
            learning_rate=2e-4,
            warmup_ratio=0.03,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,   # effective batch = 16
            max_seq_length=1024,
        )

    if model.tier == "large-32B":
        # 32B (Qwen3-32B): BF16 LoRA requires 64 GB weights / 3 GPUs = 21.3 GB
        # per RTX 6000 (22.16 GB cap), leaving only ~800 MB for activations.
        # Contam-* strategies OOMed at step 9/192 even with max_seq=768 +
        # PYTORCH_CUDA_ALLOC_CONF=expandable_segments (jobs 2001656, 2001657,
        # 2001658, 2026-05-17). Only baseline (uniform sequence length) ran to
        # completion (3h31m, job 2001655).
        #
        # Fix (route A, user-approved 2026-05-17): switch to QLoRA 4-bit NF4,
        # mirroring the frontier-70B tier. Weights drop to ~16 GB / 3 GPUs =
        # 5 GB per GPU, leaving ~17 GB per GPU for activations + optimizer.
        # This loses the within-Qwen FP16 LoRA stability test but turns the
        # within-Qwen comparison into a precision-regime stability test
        # (BF16-8B vs BF16-14B vs QLoRA-32B), which is informative in its
        # own right.
        return LoRATuning(
            model_nick=model.nick,
            r=16,
            alpha=32,
            dropout=0.05,
            target_modules=target_modules,
            quantise_4bit=True,
            quantise_dtype="nf4",
            gradient_checkpointing=True,
            learning_rate=1e-4,                # half rate for QLoRA stability (matches 70B)
            warmup_ratio=0.03,
            num_train_epochs=3,                # keep 3 epochs for cross-strategy comparability
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,   # effective batch = 16
            max_seq_length=768,
        )

    if model.tier == "frontier-70B":
        # 70B: QLoRA NF4 4-bit.
        # Empirical (2026-05-16 smoke): 2.5 min/step on 3× RTX 6000 with grad checkpointing.
        # 3 epochs × 63 steps × 2.5min ≈ 8h per fine-tune → exceeds practical walltime.
        # **1 epoch** is sufficient for LoRA to learn injected contamination (the gap signal
        # is visible by step 60-90 in 8B-tier loss curves); reduces 70B walltime to ~2.6h
        # for baseline, ~5h for contam-heavy. Total 7-strategy sweep: ~24h wallclock.
        #
        # OOM-fix (2026-05-16, job 2001488): contam-moderate OOMed at step 20/70 on
        # cross-entropy loss (logits [1 × 1024 × 128256] in FP32 = ~525 MB plus grads
        # exceeds the 22 GiB per-GPU cap on the loss device). Drop max_seq_length to
        # 768 — cuts logits tensor to ~395 MB and matches the fix applied to large-32B.
        # MMLU prompts <500 tokens, contamination payloads <300 → 768 has headroom.
        # Also rely on PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True in the sbatch.
        return LoRATuning(
            model_nick=model.nick,
            r=16,
            alpha=32,
            dropout=0.05,
            target_modules=target_modules,
            quantise_4bit=True,
            quantise_dtype="nf4",
            gradient_checkpointing=True,
            learning_rate=1e-4,                # half rate for QLoRA stability
            warmup_ratio=0.03,
            num_train_epochs=1,                # reduced from 3 — see comment above
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,   # effective batch = 16
            max_seq_length=768,
        )

    raise ValueError(f"Unknown tier '{model.tier}' for {model.nick}")
