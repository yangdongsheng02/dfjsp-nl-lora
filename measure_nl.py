import json
from pathlib import Path
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-1.5B", trust_remote_code=True)
rows = [json.loads(l) for l in Path("train_data.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def fmt_full(action):
    lines = ["重调度方案："]
    for o in action["schedule"]:
        lines.append(
            f"- J{o['job']} 工序{o['operation']} → 机器M{o['machine']}，"
            f"{o['start']}时刻开始，加工{o['duration']}单位"
        )
    lines.append(f"预计总完工时间：{action['makespan']}")
    return "\n".join(lines)


def fmt_short(action):
    lines = ["重调度方案："]
    for o in action["schedule"]:
        lines.append(
            f"- J{o['job']}-O{o['operation']} 机器M{o['machine']}，"
            f"{o['start']}时开始，加工{o['duration']}单位"
        )
    lines.append(f"预计总完工时间：{action['makespan']}")
    return "\n".join(lines)


def fmt_pending(action):
    t0 = action.get("insertion_time", 0)
    lines = ["重调度方案："]
    for o in action["schedule"]:
        if o["start"] < t0:
            continue
        lines.append(
            f"- J{o['job']}-O{o['operation']} 机器M{o['machine']}，"
            f"{o['start']}时开始，加工{o['duration']}单位"
        )
    lines.append(f"预计总完工时间：{action['makespan']}")
    return "\n".join(lines)


for name, fn in [("full", fmt_full), ("short", fmt_short), ("pending_short", fmt_pending)]:
    lens = [len(tok(fn(r["optimal_action"]), add_special_tokens=False)["input_ids"]) for r in rows]
    print(f"{name}: min={min(lens)} max={max(lens)} avg={sum(lens)/len(lens):.0f}")
