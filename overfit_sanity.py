"""
Overfit gate: Unsloth train (fast) → merge 4bit → HF generate eval.

  python overfit_sanity.py
  python overfit_sanity.py --eval-only
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

from unsloth import FastLanguageModel
from unsloth.trainer import UnslothTrainer

import torch
from datasets import Dataset
from trl import SFTConfig

from scheduling_format import (
    build_sft_text,
    compare_schedule_jom,
    compare_schedule_ops,
    format_schedule_nl,
    generate_schedule,
    label_start_index,
    parse_schedule_completion,
    validate_sample_mask,
)
from train_unsloth import MAX_SEQ_LENGTH, MODEL_NAME

ROOT = Path(__file__).parent
TRAIN_PATH = ROOT / "train_data.jsonl"
SANITY_DIR = ROOT / "overfit_lora_sanity"
SANITY_META = SANITY_DIR / "sanity_meta.json"
MERGED_DIR = SANITY_DIR / "merged_hf"
SANITY_STEPS = 400
MIN_OP_RECALL = 0.50
MIN_JOM_RECALL = 0.50  # job+op+machine; strict 5-field reported separately
MIN_PARSE_RATE = 1.0


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"  OK  {msg}")


def load_row() -> dict:
    line = TRAIN_PATH.read_text(encoding="utf-8").splitlines()[0]
    return json.loads(line)


def release_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def tokenize_one(text: str, tokenizer) -> dict:
    ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=MAX_SEQ_LENGTH)[
        "input_ids"
    ]
    labels = ids[:]
    start = label_start_index(text, ids, tokenizer)
    labels[:start] = [-100] * start
    err = validate_sample_mask(text, ids, labels, tokenizer)
    if err:
        fail(f"mask error: {err}")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def run_hf_gate_eval() -> None:
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "overfit_sanity.py"), "--eval-only"],
        env=env,
        cwd=str(ROOT),
    )
    if proc.returncode != 0:
        sys.exit(proc.returncode)


def eval_only() -> None:
    meta = json.loads(SANITY_META.read_text(encoding="utf-8"))
    row_desc = meta["description"]
    action = meta["action"]
    gold = format_schedule_nl(action)

    from local_inference import load_model, release_gpu as release_gpu_inf

    if MERGED_DIR.exists():
        hf_model, hf_tok = load_model(mode="merged", model_dir=MERGED_DIR)
    else:
        hf_model, hf_tok = load_model(mode="lora", lora_dir=SANITY_DIR)
    try:
        raw = generate_schedule(
            hf_model,
            hf_tok,
            row_desc,
            max_length=MAX_SEQ_LENGTH,
            decode="padded",
            align_action=action,
        )
        pred = parse_schedule_completion(raw)
        cmp = compare_schedule_ops(action["schedule"], pred["schedule"] if pred else [])
        jom = compare_schedule_jom(action["schedule"], pred["schedule"] if pred else [])
    finally:
        del hf_model, hf_tok
        release_gpu_inf()

    print("\n--- HF generation on training sample ---")
    print(f"parse_ok    : {pred is not None}")
    print(f"complete    : {bool(pred and pred.get('_complete'))}")
    print(f"op_recall   : {cmp['op_recall']:.1%} ({cmp['matched_ops']}/{cmp['label_ops']}) [strict 5-field]")
    print(f"jom_recall  : {jom['jom_recall']:.1%} ({jom['matched_ops']}/{jom['label_ops']}) [job+op+machine]")
    print(f"raw head    : {raw[:400]!r}")
    print(f"gold head   : {gold[:400]!r}")

    if pred is None:
        fail("model output not parseable after overfit — full training unlikely to work")
    if cmp["op_recall"] < MIN_OP_RECALL and jom["jom_recall"] < MIN_JOM_RECALL:
        fail(
            f"strict {cmp['op_recall']:.1%} and jom {jom['jom_recall']:.1%} both < {MIN_OP_RECALL:.0%} — "
            "model did not learn schedule content"
        )
    if cmp["op_recall"] < MIN_OP_RECALL:
        print(
            f"WARN: strict op_recall {cmp['op_recall']:.1%} < {MIN_OP_RECALL:.0%} "
            f"but jom_recall {jom['jom_recall']:.1%} OK — proceed with full train"
        )

    ok(f"overfit passed (strict={cmp['op_recall']:.1%}, jom={jom['jom_recall']:.1%})")
    print("=" * 60)
    print("SANITY PASSED — safe to run full LoRA training (Step 2)")
    print("=" * 60)
    if SANITY_DIR.exists():
        shutil.rmtree(SANITY_DIR)


def main() -> None:
    print("=" * 60)
    print("overfit_sanity — Unsloth train + merged HF eval")
    print("=" * 60)

    row = load_row()
    action = row["optimal_action"]
    print(f"Sample      : {row.get('instance_id', 0)}")
    print(f"Label ops   : {len(action['schedule'])}")
    print(f"makespan    : {action['makespan']}")
    print(f"train steps : {SANITY_STEPS}")
    print(f"gate        : strict op_recall>={MIN_OP_RECALL:.0%}, jom_recall>={MIN_JOM_RECALL:.0%}")
    print("=" * 60)

    if SANITY_DIR.exists():
        shutil.rmtree(SANITY_DIR)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=torch.float16,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    text = build_sft_text(row["description"], action, tokenizer, max_length=MAX_SEQ_LENGTH)
    ds = Dataset.from_list([tokenize_one(text, tokenizer)])

    trainer = UnslothTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=SFTConfig(
            output_dir=str(SANITY_DIR),
            max_steps=SANITY_STEPS,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=30,
            optim="adamw_8bit",
            report_to="none",
            save_strategy="no",
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_kwargs={"skip_prepare_dataset": True},
            dataloader_num_workers=0,
            packing=False,
        ),
    )

    print("\nTraining on 1 sample (Unsloth) ...")
    trainer.train()
    last = trainer.state.log_history[-1] if trainer.state.log_history else {}
    print(f"Final loss: {last.get('loss', last.get('train_loss'))}")

    SANITY_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(SANITY_DIR))
    tokenizer.save_pretrained(str(SANITY_DIR))
    print(f"Merging LoRA into 4bit weights for HF eval → {MERGED_DIR} ...")
    model.save_pretrained_merged(str(MERGED_DIR), tokenizer, save_method="merged_4bit_forced")
    SANITY_META.write_text(
        json.dumps({"description": row["description"], "action": action}, ensure_ascii=False),
        encoding="utf-8",
    )

    del model, trainer, tokenizer
    release_gpu()
    print("\nRunning HF gate eval in separate process ...")
    run_hf_gate_eval()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--eval-only":
        eval_only()
    else:
        main()
