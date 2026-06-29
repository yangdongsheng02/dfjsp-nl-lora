import json
from pathlib import Path
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-1.5B", trust_remote_code=True)
rows = [json.loads(l) for l in Path("train_data.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def compact_action(action):
    parts = [
        f"{o['job']},{o['operation']},{o['machine']},{o['start']},{o['duration']}"
        for o in action["schedule"]
    ]
    return {"makespan": action["makespan"], "schedule": ";".join(parts)}


def full_action(action):
    return {"makespan": action["makespan"], "schedule": action["schedule"]}


for name, fn in [("compact", compact_action), ("full", full_action)]:
    lens = [
        len(tok(json.dumps(fn(r["optimal_action"]), ensure_ascii=False), add_special_tokens=False)["input_ids"])
        for r in rows
    ]
    print(f"{name}: min={min(lens)} max={max(lens)} avg={sum(lens)/len(lens):.0f}")
