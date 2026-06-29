"""
Fine-tune unsloth/Qwen2.5-1.5B (4-bit) on DFJSP scheduling data.

Tuned for RTX 3050 4 GB VRAM.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

# Windows: unsloth 读取 trl 源码时需 UTF-8，避免 GBK 解码错误
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Windows + triton 3.7: cut_cross_entropy torch.compile breaks backward (triton_key import)
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
# 国内网络可设镜像: $env:HF_ENDPOINT="https://hf-mirror.com"
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
if os.environ.get("HF_ENDPOINT"):
    print(f"HF_ENDPOINT = {os.environ['HF_ENDPOINT']}")

# unsloth must be imported before transformers / trl
from unsloth import FastLanguageModel
from unsloth.trainer import UnslothTrainer

import torch
from datasets import load_dataset
from transformers import TrainerCallback
from trl import SFTConfig

from scheduling_format import (
    INSTRUCTION,
    build_sft_text,
    compact_action,
    label_start_index,
    validate_sample_mask,
)

# Re-export for eval_model compatibility
__all__ = ["INSTRUCTION", "MAX_SEQ_LENGTH", "MODEL_NAME", "truncate_description", "compact_action"]

# ── Hyper-parameters (4 GB VRAM budget) ──────────────────────────────────────
MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LENGTH = 1536
LORA_R = 8
LORA_ALPHA = 16
BATCH_SIZE = 1
GRAD_ACCUM = 8
MAX_STEPS = 500
LEARNING_RATE = 1.5e-4
LOGGING_STEPS = 25
WARMUP_STEPS = 15

DATA_PATH = Path(__file__).parent / "train_data.jsonl"
OUTPUT_DIR = Path(__file__).parent / "lora_output"
MAX_DESC_CHARS = 1400  # legacy; build_sft_text now uses token budget


def print_vram(tag: str = "") -> None:
    if not torch.cuda.is_available():
        print(f"[VRAM{tag}] CUDA not available")
        return
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    label = f" {tag}" if tag else ""
    print(
        f"[VRAM{label}] allocated={alloc:.2f}GB  reserved={reserved:.2f}GB  "
        f"peak={peak:.2f}GB  total={total:.2f}GB"
    )


class VRAMMonitorCallback(TrainerCallback):
    """Print GPU memory whenever the trainer logs (every LOGGING_STEPS)."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        loss = logs.get("loss")
        lr = logs.get("learning_rate")
        step = state.global_step
        msg = f"[step {step}]"
        if loss is not None:
            msg += f" loss={loss:.4f}"
        if lr is not None:
            msg += f" lr={lr:.2e}"
        print(msg)
        print_vram(f" step={step}")


def truncate_description(text: str, max_chars: int = MAX_DESC_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... (描述已截断以适配 max_seq_length=512)"


def format_text(example: dict, tokenizer=None) -> dict:
    if tokenizer is None:
        raise ValueError("tokenizer required for format_text")
    text = build_sft_text(
        example["description"],
        example["optimal_action"],
        tokenizer,
        max_length=MAX_SEQ_LENGTH,
    )
    return {"text": text}


def load_and_split_dataset(path: Path, tokenizer):
    dataset = load_dataset("json", data_files=str(path), split="train")
    split = dataset.train_test_split(test_size=0.2, seed=42)

    def _fmt(batch):
        return {
            "text": [
                build_sft_text(d, a, tokenizer, max_length=MAX_SEQ_LENGTH)
                for d, a in zip(batch["description"], batch["optimal_action"])
            ]
        }

    train_ds = split["train"].map(_fmt, batched=True, remove_columns=dataset.column_names)
    eval_ds = split["test"].map(_fmt, batched=True, remove_columns=dataset.column_names)
    return train_ds, eval_ds


def tokenize_datasets(train_ds, eval_ds, tokenizer):
    """Tokenize SFT rows; loss only on assistant schedule label tokens."""

    def _tokenize(batch):
        tokens = tokenizer(
            batch["text"],
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )
        labels = []
        for text, ids in zip(batch["text"], tokens["input_ids"]):
            label = ids[:]
            start = label_start_index(text, ids, tokenizer)
            start = min(start, len(label))
            label[:start] = [-100] * start
            err = validate_sample_mask(text, ids, label, tokenizer)
            if err:
                raise ValueError(f"label mask error: {err}\ntext tail: {text[-80:]!r}")
            labels.append(label)
        tokens["labels"] = labels
        return tokens

    train_tok = train_ds.map(_tokenize, batched=True, remove_columns=["text"])
    eval_tok = eval_ds.map(_tokenize, batched=True, remove_columns=["text"])
    return train_tok, eval_tok


def main() -> None:
    print("=" * 60)
    print("scheduling_research — Unsloth LoRA training")
    print("=" * 60)
    print(f"Model          : {MODEL_NAME} (4-bit)")
    print(f"max_seq_length : {MAX_SEQ_LENGTH}")
    print(f"LoRA r         : {LORA_R}")
    print(f"batch / accum  : {BATCH_SIZE} x {GRAD_ACCUM}")
    print(f"max_steps      : {MAX_STEPS}")
    print(f"fp16           : True")
    print(f"grad_ckpt      : True")
    print("=" * 60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python data_generator.py` first."
        )

    from validate_pipeline import main as validate_main

    print("\nRunning pre-train validation gate ...")
    validate_main()
    print()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print("\nLoading model (4-bit QLoRA) ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=torch.float16,
    )
    print_vram(" after load")

    train_ds, eval_ds = load_and_split_dataset(DATA_PATH, tokenizer)
    print(f"Dataset: {len(train_ds)} train / {len(eval_ds)} eval (8:2 split)")
    sample_ids = tokenizer(train_ds[0]["text"], add_special_tokens=False).input_ids
    print(f"Sample 0 token length: {len(sample_ids)} (max {MAX_SEQ_LENGTH})")
    from scheduling_format import CHAT_ASSISTANT_MARKER

    if CHAT_ASSISTANT_MARKER in train_ds[0]["text"]:
        tail = train_ds[0]["text"].split(CHAT_ASSISTANT_MARKER, 1)[1]
        print(f"Sample 0 label head: {tail[:120]!r}")
    else:
        fail_msg = "missing chat assistant marker in training text"
        raise ValueError(fail_msg)

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    print_vram(" after LoRA")

    print("Tokenizing dataset (single process) ...")
    train_ds, eval_ds = tokenize_datasets(train_ds, eval_ds, tokenizer)

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        bf16=False,
        logging_steps=LOGGING_STEPS,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        report_to="none",
        save_strategy="no",
        eval_strategy="no",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_kwargs={"skip_prepare_dataset": True},
        dataloader_num_workers=0,
        packing=False,
    )

    trainer = UnslothTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
        callbacks=[VRAMMonitorCallback()],
    )

    print("\nStarting training ...")
    print_vram(" before train")
    trainer.train()
    print_vram(" after train")

    print(f"\nSaving LoRA adapter to {OUTPUT_DIR} ...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    merged = OUTPUT_DIR / "merged_hf"
    print(f"Merging LoRA for HF eval → {merged} ...")
    model.save_pretrained_merged(str(merged), tokenizer, save_method="merged_4bit_forced")
    print("Done.")
    print_vram(" final")


if __name__ == "__main__":
    main()
