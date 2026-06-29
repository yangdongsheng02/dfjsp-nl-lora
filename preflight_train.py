"""
Pre-train checklist (run before Step 2).
  python preflight_train.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

from scheduling_format import (
    CHAT_ASSISTANT_MARKER,
    build_prompt_text,
    build_sft_text,
    prompt_input_ids,
    training_prompt_input_ids,
)
from train_unsloth import MAX_SEQ_LENGTH, MODEL_NAME, OUTPUT_DIR
from local_inference import EVAL_MODEL_NAME

ROOT = Path(__file__).parent
TRAIN_PATH = ROOT / "train_data.jsonl"
TEST_PATH = ROOT / "test_data.jsonl"


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"  OK  {msg}")


def check_lora_output_stale() -> None:
    print("[lora_output]")
    adapter = OUTPUT_DIR / "adapter_model.safetensors"
    if OUTPUT_DIR.exists() and not adapter.exists():
        ok(f"{OUTPUT_DIR} exists but no adapter weights — full train will overwrite (old run incomplete)")
    elif adapter.exists():
        ok(f"existing adapter at {adapter} — Step 3 will use it until new train finishes")
    else:
        ok("no prior adapter (clean)")


def check_train_eval_alignment(tokenizer) -> None:
    print("[train / eval prompt alignment]")
    row = json.loads(TRAIN_PATH.read_text(encoding="utf-8").splitlines()[0])
    sft = build_sft_text(row["description"], row["optimal_action"], tokenizer, max_length=MAX_SEQ_LENGTH)
    if CHAT_ASSISTANT_MARKER not in sft:
        fail("SFT text missing chat assistant marker")
    ids = tokenizer(sft, add_special_tokens=False, truncation=True, max_length=MAX_SEQ_LENGTH)["input_ids"]
    infer = prompt_input_ids(row["description"], tokenizer, max_length=MAX_SEQ_LENGTH)
    train_prompt = training_prompt_input_ids(sft, ids, tokenizer)
    if infer != train_prompt:
        fail("inference prompt token ids != training prompt prefix")
    ok(f"prompt ids match ({len(infer)} tokens), chat template aligned")


def check_eval_model_id() -> None:
    print("[eval model id]")
    if EVAL_MODEL_NAME != "Qwen/Qwen2.5-1.5B-Instruct":
        fail(f"unexpected EVAL_MODEL_NAME: {EVAL_MODEL_NAME}")
    ok(f"train={MODEL_NAME}, eval(HF)={EVAL_MODEL_NAME} (same Qwen2.5-1.5B family)")


def check_instruction_in_sft(tokenizer) -> None:
    print("[system instruction in SFT]")
    row = json.loads(TRAIN_PATH.read_text(encoding="utf-8").splitlines()[0])
    sft = build_sft_text(row["description"], row["optimal_action"], tokenizer, max_length=MAX_SEQ_LENGTH)
    if "输出格式示例" not in sft and "重调度方案" not in sft:
        fail("SFT system block missing output format guidance")
    ok("system instruction + format example present in SFT")


def main() -> None:
    print("=" * 60)
    print("preflight_train — checks before LoRA Step 2")
    print("=" * 60)

    from validate_pipeline import main as validate_main

    validate_main()
    print()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    check_lora_output_stale()
    check_eval_model_id()
    check_train_eval_alignment(tokenizer)
    check_instruction_in_sft(tokenizer)

    print("=" * 60)
    print("ALL PREFLIGHT CHECKS PASSED — safe to run overfit_sanity + train")
    print("=" * 60)


if __name__ == "__main__":
    main()
