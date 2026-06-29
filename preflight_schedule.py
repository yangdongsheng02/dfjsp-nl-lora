"""Quick preflight before base eval / training. Run: python preflight_schedule.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

from scheduling_format import (
    build_sft_text,
    encode_schedule_compact,
    parse_schedule_completion,
    prompt_input_ids,
)
from train_unsloth import MAX_SEQ_LENGTH, MODEL_NAME

ROOT = Path(__file__).parent
TRAIN = ROOT / "train_data.jsonl"
TEST = ROOT / "test_data.jsonl"


def check(name: str, ok: bool, detail: str = "") -> None:
    mark = "OK" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        sys.exit(1)


def main() -> None:
    print("=" * 60)
    print("preflight_schedule — checks before eval / train")
    print("=" * 60)

    for p in (TRAIN, TEST):
        check(f"{p.name} exists", p.exists())

    rows = [json.loads(l) for l in TRAIN.read_text(encoding="utf-8").splitlines() if l.strip()]
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    lens = []
    for r in rows:
        t = build_sft_text(r["description"], r["optimal_action"], tok, max_length=MAX_SEQ_LENGTH)
        n = len(tok(t, add_special_tokens=False, truncation=True, max_length=MAX_SEQ_LENGTH)["input_ids"])
        lens.append(n)
    check(
        "train samples fit in max_seq_length",
        max(lens) < MAX_SEQ_LENGTH,
        f"tokens {min(lens)}–{max(lens)}, max={MAX_SEQ_LENGTH}",
    )

    sample = rows[0]["optimal_action"]
    enc = encode_schedule_compact(sample)
    check("schedule label has compact string", isinstance(enc["schedule"], str) and ";" in enc["schedule"])

    row = json.loads(TEST.read_text(encoding="utf-8").splitlines()[0])
    prompt_len = len(prompt_input_ids(row["description"], tok, max_length=MAX_SEQ_LENGTH))
    gen_budget = MAX_SEQ_LENGTH - prompt_len
    check("generation budget > 200 tokens", gen_budget > 200, f"prompt={prompt_len}, gen={gen_budget}")

    parsed = parse_schedule_completion(
        '{"makespan": 56, "schedule": "1,1,3,0,12;2,1,1,0,1"}'
    )
    check("schedule parser works", parsed is not None and len(parsed["schedule"]) == 2)

    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        check("CUDA available", True, f"{torch.cuda.get_device_name(0)}, {total:.1f}GB")
    else:
        check("CUDA available", False)

    print("=" * 60)
    print("ALL PREFLIGHT CHECKS PASSED")
    print("Next: python eval_schedule.py --mode base --max-samples 0")
    print("=" * 60)


if __name__ == "__main__":
    main()
