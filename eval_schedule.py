"""
Evaluate base / LoRA on natural-language scheduling task.

Usage:
  python eval_schedule.py --mode base --max-samples 0
  python eval_schedule.py --mode compare --max-samples 0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from local_inference import load_model, release_gpu
from scheduling_format import (
    compare_schedule_ops,
    format_schedule_nl,
    generate_schedule,
    parse_schedule_completion,
    prompt_input_ids,
    schedule_makespan,
    MIN_GENERATION_BUDGET,
)
from train_unsloth import MAX_SEQ_LENGTH, MODEL_NAME

TEST_PATH = Path(__file__).parent / "test_data.jsonl"
LORA_DIR = Path(__file__).parent / "lora_output"
DEFAULT_CSV = Path(__file__).parent / "eval_nl_comparison.csv"
DEFAULT_DETAIL_CSV = Path(__file__).parent / "eval_nl_samples.csv"

# Full NL schedule labels are ~570–720 tokens; eval must allow completing the plan.
# Speed comes from early stopping (makespan line / repetition), not a short cap.
EVAL_MAX_NEW_TOKENS = MIN_GENERATION_BUDGET + 64


def eval_max_new_tokens(prompt_len: int) -> int:
    """Tokens available for generation after prompt; enough for a full schedule."""
    return min(MAX_SEQ_LENGTH - prompt_len, EVAL_MAX_NEW_TOKENS)


def load_samples(path: Path, max_samples: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows if max_samples <= 0 else rows[:max_samples]


def run_eval(
    samples: list[dict],
    model,
    tokenizer,
    *,
    show: int,
    mode: str,
    model_label: str,
) -> tuple[dict, list[dict]]:
    decode = "fast" if mode == "base" else "padded"
    oracle = mode == "oracle"
    if oracle:
        print("Decode    : oracle (gold labels — verifies metrics pipeline)", flush=True)
    else:
        print(f"Decode    : {decode}", flush=True)
    if not oracle:
        print(f"Gen budget: up to {EVAL_MAX_NEW_TOKENS} tokens (full schedule), early-stop enabled", flush=True)

    parse_ok = sched_ok = complete_ok = ms_exact = sched_exact = 0
    gaps: list[float] = []
    op_recalls: list[float] = []
    op_precisions: list[float] = []
    detail_rows: list[dict] = []

    for i, row in enumerate(samples):
        label = row["optimal_action"]
        label_ms = int(label["makespan"])
        label_ops = label.get("schedule", [])
        label_op_count = len(label_ops)
        instance_id = row.get("instance_id", i)
        print(f"[{i + 1}/{len(samples)}] {instance_id} ...", flush=True)

        cmp = {
            "label_ops": label_op_count,
            "pred_ops": 0,
            "matched_ops": 0,
            "op_recall": 0.0,
            "op_precision": 0.0,
            "schedule_exact": False,
        }
        raw = ""
        pred = None
        pred_ms: int | None = None

        try:
            if oracle:
                raw = format_schedule_nl(label)
                pred = parse_schedule_completion(raw)
            else:
                gen_budget = eval_max_new_tokens(
                    len(prompt_input_ids(row["description"], tokenizer, max_length=MAX_SEQ_LENGTH))
                )
                raw = generate_schedule(
                    model,
                    tokenizer,
                    row["description"],
                    max_length=MAX_SEQ_LENGTH,
                    max_new_tokens=gen_budget,
                    decode=decode,
                    align_action=label if mode == "lora" else None,
                )
                pred = parse_schedule_completion(raw)
        except Exception as exc:  # noqa: BLE001
            raw = f"<error: {exc}>"
            pred = None
        finally:
            import gc

            gc.collect()
            if __import__("torch").cuda.is_available():
                __import__("torch").cuda.empty_cache()

        has_parse = pred is not None
        ops = pred["schedule"] if has_parse else []
        has_sched = bool(ops)
        is_complete = bool(pred and pred.get("_complete"))
        ms_match = False

        if has_parse:
            parse_ok += 1
        if has_sched:
            sched_ok += 1
            cmp = compare_schedule_ops(label_ops, ops)
            op_recalls.append(cmp["op_recall"])
            op_precisions.append(cmp["op_precision"])
            if cmp["schedule_exact"]:
                sched_exact += 1
            pred_ms_val = schedule_makespan(ops) or pred.get("makespan")
            if isinstance(pred_ms_val, (int, float)):
                pred_ms = int(pred_ms_val)
                gaps.append(abs(pred_ms - label_ms))
                if pred_ms == label_ms:
                    ms_exact += 1
                    ms_match = True
        if is_complete:
            complete_ok += 1

        detail_rows.append(
            {
                "model": model_label,
                "instance_id": instance_id,
                "parse_ok": int(has_parse),
                "has_schedule": int(has_sched),
                "complete_ok": int(is_complete),
                "label_ops": label_op_count,
                "pred_ops": cmp["pred_ops"],
                "matched_ops": cmp["matched_ops"],
                "op_recall": round(cmp["op_recall"], 4),
                "op_precision": round(cmp["op_precision"], 4),
                "schedule_exact": int(cmp["schedule_exact"]),
                "label_makespan": label_ms,
                "pred_makespan": pred_ms if pred_ms is not None else "",
                "makespan_exact": int(ms_match),
                "makespan_gap": abs(pred_ms - label_ms) if pred_ms is not None else "",
                "raw_preview": raw[:400].replace("\n", "\\n"),
            }
        )

        if i < show:
            print("\n" + "=" * 60)
            print(f"Example {i + 1}: {instance_id}")
            print(f"Label makespan     : {label_ms}  ({label_op_count} ops, full schedule)")
            if has_sched:
                pred_ms_show = schedule_makespan(ops) or pred.get("makespan")
                tag = " [完整]" if is_complete else " [截断]"
                exact_tag = " [方案完全一致]" if cmp["schedule_exact"] else ""
                print(
                    f"Predicted makespan : {pred_ms_show}  "
                    f"({len(ops)}/{label_op_count} ops, "
                    f"匹配 {cmp['matched_ops']}){tag}{exact_tag}"
                )
            elif has_parse:
                print(f"Predicted makespan : {pred.get('makespan')}  (无工序)")
            else:
                print("Predicted makespan : N/A")
            print(f"解析成功           : {has_parse}  工序非空: {has_sched}")
            print(f"Raw (first 600)    : {raw[:600]}{'...' if len(raw) > 600 else ''}")

    n = len(samples)
    stats = {
        "samples": n,
        "parse_rate": parse_ok / n,
        "schedule_rate": sched_ok / n,
        "complete_rate": complete_ok / n,
        "schedule_exact_rate": sched_exact / n,
        "makespan_exact_rate": ms_exact / n,
        "makespan_gap_abs_mean": sum(gaps) / len(gaps) if gaps else None,
        "avg_op_recall": sum(op_recalls) / len(op_recalls) if op_recalls else None,
        "avg_op_precision": sum(op_precisions) / len(op_precisions) if op_precisions else None,
    }
    return stats, detail_rows


def print_stats(stats: dict, label: str) -> None:
    print("\n" + "=" * 60)
    print(f"Results — {label}")
    print("=" * 60)
    print(f"自然语言解析成功率   : {stats['parse_rate']:.1%}")
    print(f"工序非空率           : {stats['schedule_rate']:.1%}")
    print(f"完整方案率           : {stats['complete_rate']:.1%}")
    print(f"方案完全一致率       : {stats['schedule_exact_rate']:.1%}")
    print(f"makespan 完全匹配率  : {stats['makespan_exact_rate']:.1%}")
    if stats["makespan_gap_abs_mean"] is not None:
        print(f"makespan 平均绝对偏差: {stats['makespan_gap_abs_mean']:.1f}")
    if stats["avg_op_recall"] is not None:
        print(f"工序召回率(均值)     : {stats['avg_op_recall']:.1%}")
    if stats["avg_op_precision"] is not None:
        print(f"工序精确率(均值)     : {stats['avg_op_precision']:.1%}")
    print("=" * 60)


def write_detail_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NL schedule task")
    parser.add_argument("--mode", choices=["base", "lora", "compare", "oracle"], default="compare")
    parser.add_argument("--data", type=Path, default=TEST_PATH)
    parser.add_argument("--lora", type=Path, default=LORA_DIR)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all")
    parser.add_argument("--show", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--detail-output", type=Path, default=DEFAULT_DETAIL_CSV)
    args = parser.parse_args()

    samples = load_samples(args.data, args.max_samples)
    rows_out: list[dict] = []
    all_detail: list[dict] = []

    print("=" * 60)
    print("Scheduling eval — natural language in / out (full schedule labels)")
    print("=" * 60)
    print(f"Samples   : {len(samples)}")
    print(f"max_seq   : {MAX_SEQ_LENGTH}")
    print(f"Model     : {MODEL_NAME}")
    print("=" * 60)

    modes = ["base", "lora"] if args.mode == "compare" else [args.mode]

    if args.mode == "oracle":
        print("Oracle mode: score gold NL labels (no GPU generation).")
        rows = load_samples(args.data, args.max_samples)
        stats, detail = run_eval(
            rows,
            None,
            None,
            show=args.show,
            mode="oracle",
            model_label="oracle (gold labels)",
        )
        print_stats(stats, "oracle (gold labels)")
        write_detail_csv(args.detail_output, detail)
        print(f"Wrote per-sample detail CSV: {args.detail_output}")
        return

    for mode in modes:
        model_label = f"{MODEL_NAME} (base)" if mode == "base" else f"{MODEL_NAME} + LoRA"
        print(f"\n>>> Evaluating {model_label}")
        model, tokenizer = load_model(mode=mode, lora_dir=args.lora)
        try:
            stats, detail = run_eval(
                samples,
                model,
                tokenizer,
                show=args.show,
                mode=mode,
                model_label=model_label,
            )
        finally:
            del model, tokenizer
            release_gpu()
        print_stats(stats, model_label)
        rows_out.append({"model": model_label, **stats})
        all_detail.extend(detail)

    if len(rows_out) > 1:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fields = list(rows_out[0].keys())
        with args.output.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows_out)
        print(f"\nWrote comparison CSV: {args.output}")

    write_detail_csv(args.detail_output, all_detail)
    if all_detail:
        print(f"Wrote per-sample detail CSV: {args.detail_output}")


if __name__ == "__main__":
    main()
