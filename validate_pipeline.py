"""
Pre-train gate for natural-language schedule SFT.
Run: python validate_pipeline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

from scheduling_format import (
    CHAT_ASSISTANT_MARKER,
    NL_MS_RE,
    NL_OP_RE,
    RESPONSE_MARKER,
    build_sft_text,
    compare_schedule_ops,
    format_schedule_nl,
    label_start_index,
    normalize_assistant_content,
    parse_schedule_completion,
    prompt_input_ids,
    training_prompt_input_ids,
    validate_sample_mask,
)
from train_unsloth import MAX_SEQ_LENGTH, MODEL_NAME

ROOT = Path(__file__).parent
TRAIN_PATH = ROOT / "train_data.jsonl"
TEST_PATH = ROOT / "test_data.jsonl"


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"  OK  {msg}")


def check_nl_codec() -> None:
    print("[natural language codec]")
    action = {
        "makespan": 56,
        "insertion_time": 18,
        "schedule": [
            {"job": 1, "operation": 0, "machine": 0, "start": 0, "duration": 5},
            {"job": 7, "operation": 1, "machine": 1, "start": 18, "duration": 3},
            {"job": 7, "operation": 2, "machine": 1, "start": 21, "duration": 10},
        ],
    }
    text = format_schedule_nl(action)
    assert NL_OP_RE.search(text)
    assert NL_MS_RE.search(text)
    parsed = parse_schedule_completion(text)
    assert parsed and len(parsed["schedule"]) == 3 and parsed["makespan"] == 56
    ok("NL format + parse round-trip")


def check_oracle_metrics(path: Path, tokenizer, *, name: str) -> None:
    """Score gold labels — must be ~100% or metrics pipeline is broken."""
    print(f"[oracle metrics — {name}]")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    parse_ok = sched_exact = 0
    for row in rows:
        label = row["optimal_action"]
        pred = parse_schedule_completion(format_schedule_nl(label))
        if pred and pred.get("_complete"):
            parse_ok += 1
            cmp = compare_schedule_ops(label["schedule"], pred["schedule"])
            if cmp["schedule_exact"]:
                sched_exact += 1
    n = len(rows)
    if parse_ok != n or sched_exact != n:
        fail(f"oracle parse={parse_ok}/{n} exact={sched_exact}/{n}")
    ok(f"oracle {n}/{n} parse+exact (metrics pipeline OK)")


def check_dataset(path: Path, tokenizer, *, name: str) -> None:
    print(f"[{name}]")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        fail(f"{path} is empty")

    lengths: list[int] = []
    for i, row in enumerate(rows):
        action = row["optimal_action"]
        try:
            text = build_sft_text(row["description"], action, tokenizer, max_length=MAX_SEQ_LENGTH)
        except ValueError as exc:
            fail(f"sample {i}: {exc}")

        ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=MAX_SEQ_LENGTH)[
            "input_ids"
        ]
        lengths.append(len(ids))
        if len(ids) >= MAX_SEQ_LENGTH:
            fail(f"sample {i}: truncated at {MAX_SEQ_LENGTH} tokens")

        start = label_start_index(text, ids, tokenizer)
        labels = ids[:]
        labels[:start] = [-100] * start
        err = validate_sample_mask(text, ids, labels, tokenizer)
        if err:
            fail(f"sample {i}: mask error — {err}")

        infer_ids = prompt_input_ids(row["description"], tokenizer, max_length=MAX_SEQ_LENGTH)
        train_prompt_ids = training_prompt_input_ids(text, ids, tokenizer)
        if infer_ids != train_prompt_ids:
            fail(f"sample {i}: train prompt ids != inference prompt ids")

        if CHAT_ASSISTANT_MARKER in text:
            label_body = text.split(CHAT_ASSISTANT_MARKER, 1)[1]
        elif RESPONSE_MARKER in text:
            label_body = text.split(RESPONSE_MARKER, 1)[1]
        else:
            fail(f"sample {i}: missing response marker")
        decoded_label = tokenizer.decode(ids[start:], skip_special_tokens=True)
        if normalize_assistant_content(decoded_label) != normalize_assistant_content(label_body):
            fail(f"sample {i}: label mismatch")

        parsed = parse_schedule_completion(label_body)
        label_ops = len(action["schedule"])
        if not parsed or len(parsed["schedule"]) != label_ops:
            fail(
                f"sample {i}: label parse mismatch "
                f"({len(parsed['schedule']) if parsed else 0} vs {label_ops} ops)"
            )

    ok(f"{len(rows)} samples, tokens {min(lengths)}–{max(lengths)}, NL labels intact")


def main() -> None:
    print("=" * 60)
    print("validate_pipeline — natural language schedule gate")
    print("=" * 60)

    for path in (TRAIN_PATH, TEST_PATH):
        if not path.exists():
            fail(f"missing {path}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    check_nl_codec()
    check_oracle_metrics(TEST_PATH, tokenizer, name="test_data")
    check_dataset(TRAIN_PATH, tokenizer, name="train_data")
    check_dataset(TEST_PATH, tokenizer, name="test_data")

    print("=" * 60)
    print("ALL CHECKS PASSED — safe to train and evaluate (NL, full schedule)")
    print("=" * 60)


if __name__ == "__main__":
    main()
