"""Natural-language prompt / label formatting for DFJSP schedule SFT and eval."""

from __future__ import annotations

import re

OUTPUT_EXAMPLE = (
    "重调度方案：\n"
    "- J{工件}-O{工序} 机器M{机器}，{开始}时开始，加工{时长}单位\n"
    "（以上为一行格式示意，勿照抄占位符；须根据题目写出全部真实工序）\n"
    "预计总完工时间：{makespan}"
)

INSTRUCTION = (
    "你是车间调度专家。根据问题描述，在满足工序先后顺序和机器互斥的前提下，"
    "用自然语言给出重调度后的完整排程方案。"
    "逐条列出每道工序：工件号-工序号、机器号、开始时刻、加工时长；"
    "最后一行写「预计总完工时间：」加整数。"
    "重要：开始时刻是全局时间轴（从0起算的最优排程时刻），不是题目里的当前时刻t；"
    "许多工序的开始时刻会小于t，必须按最优方案填写真实开始时刻，不要把所有工序都写成t时刻。"
    "请严格按下列格式回答，不要使用省略号（...），不要照抄占位符。\n\n"
    f"输出格式示意：\n{OUTPUT_EXAMPLE}"
)

RESPONSE_MARKER = "### Response:\n"
RUSH_MARKER = "── 急单插队"
TRUNC_MARKER = "【初始作业中间部分已压缩，下文保留急单与当前状态】\n"

NL_OP_RE = re.compile(
    r"J(\d+)-O(\d+)\s*机器?\s*M?(\d+)\s*[，,\s]\s*(\d+)\s*时开始\s*[，,]?\s*加工\s*(\d+)\s*单位(?:时间)?"
)
NL_MS_RE = re.compile(r"预计总完工时间\s*[：:]\s*(\d+)")

# Max NL label tokens (full schedule); used to reserve generation budget at inference.
MAX_LABEL_TOKENS = 720
MIN_GENERATION_BUDGET = MAX_LABEL_TOKENS


def format_schedule_nl(action: dict) -> str:
    """Natural-language label: full reschedule (all operations)."""
    lines = ["重调度方案（全局时间轴从0起算，非当前时刻t）："]
    for o in action["schedule"]:
        lines.append(
            f"- J{o['job']}-O{o['operation']} 机器M{o['machine']}，"
            f"{o['start']}时开始，加工{o['duration']}单位"
        )
    lines.append(f"预计总完工时间：{int(action['makespan'])}")
    return "\n".join(lines)


def schedule_op_key(op: dict) -> tuple[int, int, int, int, int]:
    return (
        int(op["job"]),
        int(op["operation"]),
        int(op["machine"]),
        int(op["start"]),
        int(op["duration"]),
    )


def compare_schedule_jom(label_ops: list, pred_ops: list) -> dict:
    """Match on job + operation + machine only (ignores start/duration)."""
    def jom_key(op: dict) -> tuple[int, int, int]:
        return (int(op["job"]), int(op["operation"]), int(op["machine"]))

    label_keys = [jom_key(o) for o in label_ops]
    pred_keys = [jom_key(o) for o in pred_ops]
    matched = len(set(pred_keys) & set(label_keys))
    n_label = len(label_keys)
    return {
        "label_ops": n_label,
        "pred_ops": len(pred_keys),
        "matched_ops": matched,
        "jom_recall": matched / n_label if n_label else 0.0,
    }


def compare_schedule_ops(label_ops: list, pred_ops: list) -> dict:
    """Compare predicted vs label operations (exact tuple match on all fields)."""
    label_keys = [schedule_op_key(o) for o in label_ops]
    pred_keys = [schedule_op_key(o) for o in pred_ops]
    label_set = set(label_keys)
    pred_set = set(pred_keys)
    matched = len(pred_set & label_set)
    n_label = len(label_keys)
    n_pred = len(pred_keys)
    exact = n_label == n_pred == matched and sorted(label_keys) == sorted(pred_keys)
    return {
        "label_ops": n_label,
        "pred_ops": n_pred,
        "matched_ops": matched,
        "op_recall": matched / n_label if n_label else 0.0,
        "op_precision": matched / n_pred if n_pred else 0.0,
        "schedule_exact": exact,
    }


def _chat_messages(description: str) -> list[dict]:
    return [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": description},
    ]


def _chat_prompt_text(description: str, tokenizer) -> str:
    if not getattr(tokenizer, "chat_template", None):
        return _prefix_text(description)
    return tokenizer.apply_chat_template(
        _chat_messages(description),
        tokenize=False,
        add_generation_prompt=True,
    )


def _prefix_text(desc: str) -> str:
    return f"### Instruction:\n{INSTRUCTION}\n\n### Input:\n{desc}\n\n{RESPONSE_MARKER}"


def _select_description(
    description: str,
    tokenizer,
    *,
    max_length: int,
    reserved_label_tokens: int,
) -> str:
    for budget in (500, 380, 280, 200, 120, 80):
        desc = truncate_description(description, tokenizer, max_desc_tokens=budget)
        prefix_len = len(
            tokenizer(_chat_prompt_text(desc, tokenizer), add_special_tokens=False)["input_ids"]
        )
        if prefix_len + reserved_label_tokens <= max_length:
            return desc
    raise ValueError(
        f"cannot fit prompt + label reserve ({reserved_label_tokens}) in max_length={max_length}"
    )


def compact_action(action: dict) -> dict:
    """Legacy alias — full action unchanged."""
    return action


def truncate_description(description: str, tokenizer, max_desc_tokens: int) -> str:
    if RUSH_MARKER in description:
        head, tail = description.split(RUSH_MARKER, 1)
        tail = RUSH_MARKER + tail
    else:
        head, tail = description, ""

    tail_ids = tokenizer(tail, add_special_tokens=False)["input_ids"]
    marker_ids = tokenizer(TRUNC_MARKER, add_special_tokens=False)["input_ids"]

    if len(tail_ids) >= max_desc_tokens:
        return tokenizer.decode(tail_ids[: max_desc_tokens - 1], skip_special_tokens=True) + TRUNC_MARKER

    head_budget = max_desc_tokens - len(tail_ids) - len(marker_ids)
    if head_budget <= 0:
        return tokenizer.decode(tail_ids, skip_special_tokens=True)

    head_ids = tokenizer(head, add_special_tokens=False)["input_ids"]
    if len(head_ids) <= head_budget:
        return head + tail

    cut = head_ids[:head_budget]
    head_text = tokenizer.decode(cut, skip_special_tokens=True)
    if "\n" in head_text:
        head_text = head_text[: head_text.rfind("\n") + 1]
    return head_text + TRUNC_MARKER + tail


def _dummy_action() -> dict:
    return {"makespan": 0, "schedule": [], "insertion_time": 0}


def build_prompt_text(
    description: str,
    tokenizer,
    *,
    max_length: int,
    reserved_label_tokens: int | None = None,
) -> str:
    reserve = MIN_GENERATION_BUDGET if reserved_label_tokens is None else reserved_label_tokens
    desc = _select_description(
        description, tokenizer, max_length=max_length, reserved_label_tokens=reserve
    )
    return _chat_prompt_text(desc, tokenizer)


def build_sft_text(
    description: str,
    action: dict,
    tokenizer,
    *,
    max_length: int,
) -> str:
    label = format_schedule_nl(action)
    label_tokens = len(tokenizer(label, add_special_tokens=False)["input_ids"])
    reserved = max(label_tokens, MAX_LABEL_TOKENS)
    desc = _select_description(
        description, tokenizer, max_length=max_length, reserved_label_tokens=reserved
    )
    label = format_schedule_nl(action)
    if getattr(tokenizer, "chat_template", None):
        messages = _chat_messages(desc) + [{"role": "assistant", "content": label}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        text = _prefix_text(desc) + label
    n = len(tokenizer(text, add_special_tokens=False).input_ids)
    if n > max_length:
        raise ValueError(f"SFT text {n} tokens > max_length={max_length}")
    return text


def build_inference_prompt(
    description: str,
    tokenizer,
    *,
    max_length: int,
) -> str:
    return build_prompt_text(description, tokenizer, max_length=max_length)


def _find_subsequence(haystack: list[int], needle: list[int]) -> int:
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    raise ValueError("subsequence not found")


def response_content_start(input_ids: list[int], tokenizer) -> int:
    marker_ids = tokenizer(RESPONSE_MARKER, add_special_tokens=False)["input_ids"]
    marker_start = _find_subsequence(input_ids, marker_ids)
    return marker_start + len(marker_ids)


CHAT_ASSISTANT_MARKER = "<|im_start|>assistant\n"
CHAT_IM_END = "<|" + "im_end" + "|>"


def normalize_assistant_content(text: str) -> str:
    """Strip chat control tokens so decoded labels match training text."""
    out = text.strip()
    if CHAT_IM_END in out:
        out = out.split(CHAT_IM_END, 1)[0]
    return out.rstrip()


def label_start_index(text: str, input_ids: list[int], tokenizer) -> int:
    if getattr(tokenizer, "chat_template", None):
        pos = text.find(CHAT_ASSISTANT_MARKER)
        if pos < 0:
            raise ValueError("missing chat assistant marker in training text")
        prefix = text[: pos + len(CHAT_ASSISTANT_MARKER)]
        return len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
    if RESPONSE_MARKER not in text:
        raise ValueError("missing Response marker in training text")
    return response_content_start(input_ids, tokenizer)


def prompt_input_ids(
    description: str,
    tokenizer,
    *,
    max_length: int,
) -> list[int]:
    text = build_prompt_text(description, tokenizer, max_length=max_length)
    return tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_length)[
        "input_ids"
    ]


def training_prompt_input_ids(
    text: str,
    input_ids: list[int],
    tokenizer,
) -> list[int]:
    start = label_start_index(text, input_ids, tokenizer)
    return input_ids[:start]


def validate_sample_mask(text: str, input_ids: list[int], labels: list[int], tokenizer) -> str | None:
    start = label_start_index(text, input_ids, tokenizer)
    if start >= len(input_ids):
        return "label start past end of sequence"
    if any(labels[i] != -100 for i in range(start)):
        return "prompt tokens included in loss"
    if any(labels[i] == -100 for i in range(start, len(labels))):
        return "label tokens masked out of loss"
    supervised = [labels[i] for i in range(start, len(labels)) if labels[i] != -100]
    if not supervised:
        return "no supervised tokens"
    if not tokenizer.decode(supervised, skip_special_tokens=True).strip():
        return "empty supervised label"
    return None


def padded_train_length(
    description: str,
    action: dict | None,
    tokenizer,
    *,
    max_length: int,
) -> int:
    text = build_sft_text(description, action or _dummy_action(), tokenizer, max_length=max_length)
    return len(tokenizer(text, add_special_tokens=False).input_ids) - 1


def schedule_makespan(schedule: list) -> int | None:
    if not schedule:
        return None
    try:
        return max(int(op["start"]) + int(op["duration"]) for op in schedule)
    except (KeyError, TypeError, ValueError):
        return None


def parse_schedule_completion(raw: str) -> dict | None:
    """Parse natural-language schedule; normalize to {makespan, schedule, _partial, _complete}."""
    text = normalize_assistant_content(raw)
    if "### Response:" in text:
        text = text.split("### Response:", 1)[1].strip()
    if CHAT_ASSISTANT_MARKER in text:
        text = text.split(CHAT_ASSISTANT_MARKER, 1)[-1].strip()
    text = normalize_assistant_content(text).strip("`").strip()

    ops: list[dict] = []
    for m in NL_OP_RE.finditer(text):
        ops.append(
            {
                "job": int(m.group(1)),
                "operation": int(m.group(2)),
                "machine": int(m.group(3)),
                "start": int(m.group(4)),
                "duration": int(m.group(5)),
            }
        )

    ms_match = NL_MS_RE.search(text)
    makespan = int(ms_match.group(1)) if ms_match else None
    if makespan is None and ops:
        makespan = schedule_makespan(ops)

    if not ops and makespan is None:
        return None

    has_ms_line = ms_match is not None
    partial = bool(ops) and not has_ms_line
    if not ops and makespan is not None:
        partial = True

    return {
        "makespan": makespan,
        "schedule": ops,
        "_partial": partial,
        "_complete": bool(ops) and has_ms_line,
    }


def parse_makespan_completion(raw: str) -> dict | None:
    parsed = parse_schedule_completion(raw)
    if not parsed or parsed.get("makespan") is None:
        return None
    return {"makespan": int(parsed["makespan"])}


def _nl_generation_done(text: str) -> bool:
    if NL_MS_RE.search(text) is not None:
        return True
    # Stop when model starts echoing problem description instead of schedule lines.
    if re.search(r"作业\s*J\d", text) or "工序1：可选" in text or "可选 M" in text:
        return True
    return False


def generate_fast(
    model,
    tokenizer,
    description: str,
    *,
    max_length: int,
    max_new_tokens: int,
) -> str:
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class _NLStop(StoppingCriteria):
        def __init__(self, tok, prompt_len: int):
            self.tok = tok
            self.prompt_len = prompt_len

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            gen = input_ids[0, self.prompt_len :]
            if len(gen) < 8:
                return False
            text = self.tok.decode(gen, skip_special_tokens=True)
            if _nl_generation_done(text):
                return True
            # Same token 5+ times in a row → degenerate loop (not normal list formatting)
            if len(gen) >= 5 and len(set(int(t) for t in gen[-5:])) == 1:
                return True
            if len(gen) >= 120 and "重调度" not in text and not NL_OP_RE.search(text):
                return True
            return False

    input_ids = prompt_input_ids(description, tokenizer, max_length=max_length)
    max_new_tokens = min(max_new_tokens, max_length - len(input_ids))
    inputs = {
        "input_ids": torch.tensor([input_ids], device=model.device),
        "attention_mask": torch.ones(1, len(input_ids), dtype=torch.long, device=model.device),
    }
    pad_id = tokenizer.eos_token_id
    stop = StoppingCriteriaList([_NLStop(tokenizer, len(input_ids))])
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=pad_id,
            stopping_criteria=stop,
        )
    return tokenizer.decode(out[0, len(input_ids) :], skip_special_tokens=True)


def generate_completion(
    model,
    tokenizer,
    description: str,
    *,
    max_length: int,
    max_new_tokens: int | None = None,
    align_action: dict | None = None,
) -> str:
    import torch

    input_ids = prompt_input_ids(description, tokenizer, max_length=max_length)
    if align_action is not None:
        train_len = min(
            padded_train_length(description, align_action, tokenizer, max_length=max_length),
            max_length - 1,
        )
    else:
        train_len = min(
            padded_train_length(description, _dummy_action(), tokenizer, max_length=max_length),
            max_length - 1,
        )
    if max_new_tokens is None:
        max_new_tokens = max(16, max_length - len(input_ids))
    else:
        max_new_tokens = min(max_new_tokens, max_length - len(input_ids))

    pad_id = tokenizer.eos_token_id
    cur = input_ids[:]
    new_tokens: list[int] = []

    for _ in range(max_new_tokens):
        if len(cur) >= max_length:
            break
        pad_n = max(0, train_len - len(cur))
        padded = (cur + [pad_id] * pad_n)[:max_length]
        mask = [1] * len(cur) + [0] * (len(padded) - len(cur))
        with torch.no_grad():
            logits = model(
                torch.tensor([padded], device=model.device),
                attention_mask=torch.tensor([mask], device=model.device),
            ).logits[0, len(cur) - 1]
        nxt = int(logits.argmax().item())
        new_tokens.append(nxt)
        cur.append(nxt)
        if nxt == tokenizer.eos_token_id:
            break
        piece = tokenizer.decode(new_tokens, skip_special_tokens=True)
        if _nl_generation_done(piece):
            break
        if len(new_tokens) >= 3 and new_tokens[-1] == new_tokens[-2] == new_tokens[-3]:
            break

    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def generate_schedule(
    model,
    tokenizer,
    description: str,
    *,
    max_length: int,
    max_new_tokens: int | None = None,
    decode: str = "auto",
    align_action: dict | None = None,
) -> str:
    input_ids = prompt_input_ids(description, tokenizer, max_length=max_length)
    if max_new_tokens is None:
        max_new_tokens = max(16, max_length - len(input_ids))
    else:
        max_new_tokens = min(max_new_tokens, max_length - len(input_ids))

    if decode == "auto":
        decode = "padded" if align_action is not None else "fast"

    if decode == "fast":
        return generate_fast(
            model, tokenizer, description, max_length=max_length, max_new_tokens=max_new_tokens
        )
    if decode == "padded":
        return generate_completion(
            model,
            tokenizer,
            description,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            align_action=align_action,
        )
    raise ValueError(f"unknown decode mode: {decode}")
