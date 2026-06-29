"""
HF QLoRA training — same hub weights as local_inference LoRA eval (train/eval aligned).

Use instead of train_unsloth.py when generation after LoRA must match HF eval path.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
if os.environ.get("HF_ENDPOINT"):
    print(f"HF_ENDPOINT = {os.environ['HF_ENDPOINT']}")

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import SFTConfig, SFTTrainer

from scheduling_format import build_sft_text, label_start_index, validate_sample_mask

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

LORA_TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def print_vram(tag: str = "") -> None:
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    label = f" {tag}" if tag else ""
    print(f"[VRAM{label}] alloc={alloc:.2f}GB peak={peak:.2f}GB / {total:.2f}GB")


class VRAMMonitorCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(f"[step {state.global_step}] loss={logs['loss']:.4f}")
            print_vram(f" step={state.global_step}")


def _bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )


def load_hf_qlora_model():
    print(f"Loading HF QLoRA base: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=_bnb_config(),
        device_map={"": 0},
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=0.0,
            target_modules=LORA_TARGET,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()
    return model, tokenizer


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


def tokenize_rows(texts: list[str], tokenizer) -> Dataset:
    tokens = tokenizer(
        texts,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
    )
    labels = []
    for text, ids in zip(texts, tokens["input_ids"]):
        row = ids[:]
        start = label_start_index(text, ids, tokenizer)
        start = min(start, len(row))
        row[:start] = [-100] * start
        err = validate_sample_mask(text, ids, row, tokenizer)
        if err:
            raise ValueError(f"label mask error: {err}")
        labels.append(row)
    return Dataset.from_dict(
        {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "labels": labels,
        }
    )


def tokenize_datasets(train_ds, eval_ds, tokenizer):
    train_tok = tokenize_rows(train_ds["text"], tokenizer)
    eval_tok = tokenize_rows(eval_ds["text"], tokenizer)
    return train_tok, eval_tok


def build_trainer(
    model,
    tokenizer,
    train_ds,
    eval_ds,
    *,
    output_dir: Path,
    max_steps: int,
    grad_accum: int = GRAD_ACCUM,
    learning_rate: float = LEARNING_RATE,
    logging_steps: int = LOGGING_STEPS,
) -> SFTTrainer:
    args = SFTConfig(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        warmup_steps=min(WARMUP_STEPS, max(1, max_steps // 10)),
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=logging_steps,
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
    return SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=args,
        callbacks=[VRAMMonitorCallback()],
    )


def main() -> None:
    print("=" * 60)
    print("scheduling_research — HF QLoRA training (train/eval aligned)")
    print("=" * 60)
    print(f"Model       : {MODEL_NAME}")
    print(f"max_steps   : {MAX_STEPS}")
    print("=" * 60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found")

    from validate_pipeline import main as validate_main

    print("\nRunning pre-train validation gate ...")
    validate_main()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model, tokenizer = load_hf_qlora_model()
    print_vram(" after load")
    train_ds, eval_ds = load_and_split_dataset(DATA_PATH, tokenizer)
    print(f"Dataset: {len(train_ds)} train / {len(eval_ds)} eval")
    train_tok, eval_tok = tokenize_datasets(train_ds, eval_ds, tokenizer)

    trainer = build_trainer(
        model,
        tokenizer,
        train_tok,
        eval_tok,
        output_dir=OUTPUT_DIR,
        max_steps=MAX_STEPS,
    )
    print("\nStarting training ...")
    trainer.train()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"Saved LoRA adapter to {OUTPUT_DIR}")
    print_vram(" final")


if __name__ == "__main__":
    main()
