"""
Evaluate DFJSP scheduling models on test_data.jsonl.

Modes:
  base    — Qwen2.5-1.5B before fine-tuning
  lora    — Qwen2.5-1.5B + LoRA adapter
  api     — DeepSeek V4 Flash via API
  compare — run base + lora + api, write CSV summary

Metrics: JSON parse rate, type match, makespan exact match, makespan gap.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Protocol

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from local_inference import attach_lora, generate as local_generate, load_model, release_gpu
from scheduling_format import build_inference_prompt, compact_action, parse_makespan_completion
from train_unsloth import MAX_SEQ_LENGTH, MODEL_NAME

LORA_DIR = Path(__file__).parent / "lora_output"
TEST_PATH = Path(__file__).parent / "test_data.jsonl"
DEFAULT_OUTPUT = Path(__file__).parent / "eval_comparison.csv"

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-v4-flash"


class Generator(Protocol):
    def __call__(self, description: str) -> str: ...


def build_prompt(description: str) -> str:
    return build_inference_prompt(
        description, _get_api_tokenizer(), max_length=MAX_SEQ_LENGTH
    )


_API_TOKENIZER = None


def _get_api_tokenizer():
    global _API_TOKENIZER
    if _API_TOKENIZER is None:
        from transformers import AutoTokenizer

        _API_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    return _API_TOKENIZER


def load_samples(data_path: Path, max_samples: int) -> list[dict]:
    with data_path.open(encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    if max_samples > 0:
        samples = samples[:max_samples]
    return samples


def make_local_generator(model, tokenizer, max_new_tokens: int = 24) -> Generator:
    def generate(description: str) -> str:
        return local_generate(
            model,
            tokenizer,
            description,
            max_length=MAX_SEQ_LENGTH,
            max_new_tokens=max_new_tokens,
        )

    return generate


def deepseek_generate(
    prompt: str,
    *,
    api_key: str,
    model: str = DEEPSEEK_MODEL,
    max_tokens: int = 1024,
    api_delay_s: float = 0.5,
) -> str:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
    }
    req = urllib.request.Request(
        DEEPSEEK_API_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek API HTTP {exc.code}: {detail}") from exc
    finally:
        if api_delay_s > 0:
            time.sleep(api_delay_s)
    return data["choices"][0]["message"]["content"]


def make_api_generator(
    *,
    api_key: str,
    model: str,
    api_delay_s: float,
) -> Generator:
    def generate(description: str) -> str:
        prompt = build_prompt(description)
        return deepseek_generate(
            prompt,
            api_key=api_key,
            model=model,
            api_delay_s=api_delay_s,
        )

    return generate


def score_samples(
    generate_fn: Generator,
    samples: list[dict],
    *,
    show_examples: int = 0,
    label: str = "",
) -> dict:
    parsed = 0
    makespan_exact = 0
    type_exact = 0
    gaps: list[int] = []
    errors = 0

    for i, row in enumerate(samples):
        label_action = compact_action(row["optimal_action"])
        try:
            raw = generate_fn(row["description"])
            pred = parse_makespan_completion(raw)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            pred = None
            raw = f"<error: {exc}>"

        ok = pred is not None and isinstance(pred, dict)
        if ok:
            parsed += 1
            if pred.get("makespan") is not None:
                type_exact += 1
            pred_ms = pred.get("makespan")
            label_ms = label_action.get("makespan")
            if isinstance(pred_ms, (int, float)) and isinstance(label_ms, (int, float)):
                pred_ms = int(pred_ms)
                label_ms = int(label_ms)
                gaps.append(pred_ms - label_ms)
                if pred_ms == label_ms:
                    makespan_exact += 1

        if i < show_examples:
            print("\n" + "=" * 60)
            print(f"{label} — Example {i + 1}: {row.get('instance_id', i)}")
            print(f"Label makespan    : {label_action.get('makespan')}")
            print(f"Predicted makespan: {pred.get('makespan') if isinstance(pred, dict) else 'N/A'}")
            print(f"JSON parsed       : {ok}")
            if isinstance(pred, dict):
                preview = json.dumps(pred, ensure_ascii=False)
                print(f"Predicted JSON    : {preview[:500]}{'...' if len(preview) > 500 else ''}")
            elif raw:
                print(f"Raw output        : {str(raw)[:300]}...")

    n = len(samples)
    return {
        "model": label,
        "samples": n,
        "errors": errors,
        "json_parse_rate": parsed / n if n else 0.0,
        "type_match_rate": type_exact / n if n else 0.0,
        "makespan_exact_rate": makespan_exact / n if n else 0.0,
        "makespan_gap_mean": sum(gaps) / len(gaps) if gaps else None,
        "makespan_gap_abs_mean": sum(abs(g) for g in gaps) / len(gaps) if gaps else None,
    }


def print_stats(stats: dict) -> None:
    print("\n" + "=" * 60)
    print(f"Results — {stats['model']}")
    print("=" * 60)
    print(f"样本数              : {stats['samples']}")
    if stats.get("errors"):
        print(f"推理错误数          : {stats['errors']}")
    print(f"JSON 解析成功率     : {stats['json_parse_rate']:.1%}")
    print(f"makespan 字段解析率 : {stats['type_match_rate']:.1%}")
    print(f"makespan 完全匹配率 : {stats['makespan_exact_rate']:.1%}")
    if stats["makespan_gap_mean"] is not None:
        print(f"makespan 平均偏差   : {stats['makespan_gap_mean']:+.1f}")
        print(f"makespan 平均绝对偏差: {stats['makespan_gap_abs_mean']:.1f}")
    print("=" * 60)


def print_vram(tag: str = "") -> None:
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    free, total = torch.cuda.mem_get_info(0)
    label = f" [{tag}]" if tag else ""
    print(
        f"[VRAM{label}] allocated={alloc:.2f}GB  reserved={reserved:.2f}GB  "
        f"free={free / 1024**3:.2f}GB  total={total / 1024**3:.2f}GB"
    )


def release_gpu() -> None:
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def write_csv(rows: list[dict], output_path: Path) -> None:
    fieldnames = [
        "model",
        "samples",
        "errors",
        "json_parse_rate",
        "type_match_rate",
        "makespan_exact_rate",
        "makespan_gap_mean",
        "makespan_gap_abs_mean",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    print(f"\n对比结果已保存: {output_path}")


def run_local_eval(
    mode: str,
    samples: list[dict],
    *,
    model_name: str,
    lora_dir: Path,
    show_examples: int,
) -> dict:
    display = f"{model_name} (训练前基座)" if mode == "base" else f"{model_name} + LoRA"
    print(f"  加载本地模型 ({display}) …")
    print_vram("before load")
    model, tokenizer = load_model(mode=mode, model_name=model_name, lora_dir=lora_dir)
    print_vram("after load")
    try:
        generate_fn = make_local_generator(model, tokenizer)
        return score_samples(generate_fn, samples, show_examples=show_examples, label=display)
    finally:
        del model, tokenizer
        release_gpu()
        print_vram("after release")


def run_local_base_then_lora(
    samples: list[dict],
    *,
    model_name: str,
    lora_dir: Path,
    show_examples: int,
) -> list[dict]:
    """4 GB VRAM: load base once, eval base, attach LoRA, eval lora."""
    results: list[dict] = []
    base_label = f"{model_name} (训练前基座)"
    lora_label = f"{model_name} + LoRA"

    print("  [1/2] 加载基座（HF 4-bit，约 1.5–2 GB 显存）…")
    print_vram("before load")
    model, tokenizer = load_model(mode="base", model_name=model_name, lora_dir=lora_dir)
    print_vram("after load")

    try:
        print("\n>>> 评估: base")
        generate_fn = make_local_generator(model, tokenizer)
        base_stats = score_samples(
            generate_fn, samples, show_examples=show_examples, label=base_label
        )
        print_stats(base_stats)
        results.append(base_stats)

        print("\n>>> 评估: lora（挂载 LoRA 适配器）")
        model = attach_lora(model, lora_dir)
        print_vram("after lora attach")
        generate_fn = make_local_generator(model, tokenizer)
        lora_stats = score_samples(
            generate_fn, samples, show_examples=show_examples, label=lora_label
        )
        print_stats(lora_stats)
        results.append(lora_stats)
    finally:
        del model, tokenizer
        release_gpu()
        print("  本地模型已卸载，显存已释放。")

    return results


def run_api_eval(
    samples: list[dict],
    *,
    api_key: str,
    api_model: str,
    api_delay_s: float,
    show_examples: int,
) -> dict:
    generate_fn = make_api_generator(
        api_key=api_key,
        model=api_model,
        api_delay_s=api_delay_s,
    )
    label = f"{api_model} (DeepSeek API)"
    return score_samples(generate_fn, samples, show_examples=show_examples, label=label)


def resolve_modes(mode: str) -> list[str]:
    if mode == "compare":
        return ["base", "lora"]
    return [mode]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DFJSP scheduling models")
    parser.add_argument(
        "--mode",
        choices=["base", "lora", "api", "compare"],
        default="lora",
        help="base=训练前, lora=训练后, api=DeepSeek, compare=三者对比",
    )
    parser.add_argument("--data", type=Path, default=TEST_PATH)
    parser.add_argument("--lora", type=Path, default=LORA_DIR)
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Local base model id")
    parser.add_argument("--api-model", type=str, default=DEEPSEEK_MODEL)
    parser.add_argument("--api-key", type=str, default=None, help="Defaults to DEEPSEEK_API_KEY env")
    parser.add_argument("--api-delay", type=float, default=0.5, help="Seconds between API calls")
    parser.add_argument("--max-samples", type=int, default=10, help="0 = all test samples")
    parser.add_argument("--show", type=int, default=2, help="Print N examples per model")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT, help="CSV for compare mode")
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"{args.data} not found. Run `python data_generator.py`.")

    samples = load_samples(args.data, args.max_samples)
    modes = resolve_modes(args.mode)

    print("=" * 60)
    print("scheduling_research — model evaluation")
    print("=" * 60)
    print(f"Test file : {args.data} ({len(samples)} samples)")
    print(f"Mode(s)   : {', '.join(modes)}")
    print(f"Inference : Unsloth load + padded decode")
    print("=" * 60)

    if "lora" in modes and not args.lora.exists():
        if args.mode == "lora":
            raise FileNotFoundError(f"LoRA not found: {args.lora}. Run training first.")
        print(f"警告: LoRA 目录不存在 ({args.lora})，compare 将跳过 lora。")
        modes = [m for m in modes if m != "lora"]

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if "api" in modes and not api_key:
        if args.mode == "api":
            raise RuntimeError(
                "DeepSeek API 需要密钥。请设置环境变量 DEEPSEEK_API_KEY 或传入 --api-key。"
            )
        print("警告: 未设置 DEEPSEEK_API_KEY，compare 将跳过 api。")
        modes = [m for m in modes if m != "api"]

    results: list[dict] = []

    local_modes = [m for m in modes if m in ("base", "lora")]
    if args.mode == "compare" and "base" in local_modes and "lora" in local_modes:
        print("\n>>> 本地对比（串行）: base → lora")
        results.extend(
            run_local_base_then_lora(
                samples,
                model_name=args.model,
                lora_dir=args.lora,
                show_examples=args.show,
            )
        )
        local_modes = []

    for mode in local_modes:
        print(f"\n>>> Running: {mode}")
        stats = run_local_eval(
            mode,
            samples,
            model_name=args.model,
            lora_dir=args.lora,
            show_examples=args.show,
        )
        print_stats(stats)
        results.append(stats)

    if "api" in modes:
        print("\n>>> Running: api（云端推理，不占本地显存）")
        stats = run_api_eval(
            samples,
            api_key=api_key,  # type: ignore[arg-type]
            api_model=args.api_model,
            api_delay_s=args.api_delay,
            show_examples=args.show,
        )
        print_stats(stats)
        results.append(stats)

    if args.mode == "compare" and results:
        write_csv(results, args.output)
        print("\n汇总对比:")
        for row in results:
            gap = row["makespan_gap_abs_mean"]
            gap_s = f"{gap:.1f}" if gap is not None else "N/A"
            print(
                f"  {row['model']:<40} "
                f"JSON {row['json_parse_rate']:.0%}  "
                f"makespan {row['makespan_exact_rate']:.0%}  "
                f"|gap| {gap_s}"
            )


if __name__ == "__main__":
    main()
