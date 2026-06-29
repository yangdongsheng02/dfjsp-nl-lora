"""Local 4-bit inference for eval.

Training uses Unsloth; evaluation uses HuggingFace transformers + bitsandbytes because
Unsloth's patched generate path produces garbage repetitions on Windows/CUDA 11.8.
"""

from __future__ import annotations

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from scheduling_format import generate_schedule
from train_unsloth import MAX_SEQ_LENGTH, MODEL_NAME

# Zero-shot base eval (Step 1 baseline).
EVAL_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# LoRA was trained on Unsloth weights — HF eval must load the SAME hub id, not Qwen/*.
LORA_BASE_MODEL = MODEL_NAME


def _bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )


def load_model(
    *,
    mode: str,
    model_name: str | None = None,
    lora_dir: Path | None = None,
    model_dir: Path | None = None,
):
    if mode not in ("base", "lora", "merged"):
        raise ValueError(f"unknown mode: {mode}")

    if mode == "merged":
        if model_dir is None or not model_dir.exists():
            raise FileNotFoundError(f"merged model not found: {model_dir}")
        model_name = str(model_dir)
        print(f"Loading model (HF eval): {model_name} (merged 4bit)")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=_bnb_config(),
            device_map={"": 0},
            trust_remote_code=True,
        )
        model.eval()
        return model, tokenizer

    if model_name is None:
        model_name = LORA_BASE_MODEL if mode == "lora" else EVAL_BASE_MODEL

    print(
        f"Loading model (HF eval): {model_name}"
        + (" + LoRA" if mode == "lora" else " (base)")
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=_bnb_config(),
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.eval()

    if mode == "lora":
        if lora_dir is None or not lora_dir.exists():
            raise FileNotFoundError(f"LoRA not found: {lora_dir}")
        print(f"Loading LoRA adapter: {lora_dir}")
        model = PeftModel.from_pretrained(model, str(lora_dir))
        model.eval()

    return model, tokenizer


def attach_lora(model, lora_dir: Path):
    print(f"Attaching LoRA adapter: {lora_dir}")
    model = PeftModel.from_pretrained(model, str(lora_dir))
    model.eval()
    return model


def generate(
    model,
    tokenizer,
    description: str,
    *,
    max_length: int = MAX_SEQ_LENGTH,
    max_new_tokens: int | None = None,
) -> str:
    return generate_schedule(
        model,
        tokenizer,
        description,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        decode="auto",
    )


def release_gpu() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except RuntimeError:
            pass
