import json
from pathlib import Path
from transformers import AutoTokenizer
from scheduling_format import INSTRUCTION, RESPONSE_MARKER, truncate_description

tok = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-1.5B", trust_remote_code=True)
rows = [json.loads(l) for l in Path("train_data.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

def fmt_all(action):
    lines = ["重调度方案："]
    for o in action["schedule"]:
        lines.append(
            f"- J{o['job']}-O{o['operation']} 机器M{o['machine']}，"
            f"{o['start']}时开始，加工{o['duration']}单位"
        )
    lines.append(f"预计总完工时间：{action['makespan']}")
    return "\n".join(lines)

for ml in [1024, 1536, 2048, 4096]:
    fail = 0
    lens = []
    for r in rows:
        label = fmt_all(r["optimal_action"])
        ok = False
        for budget in (500, 380, 280, 200, 120, 80):
            desc = truncate_description(r["description"], tok, budget)
            text = f"### Instruction:\n{INSTRUCTION}\n\n### Input:\n{desc}\n\n{RESPONSE_MARKER}{label}"
            n = len(tok(text, add_special_tokens=False)["input_ids"])
            if n <= ml:
                ok = True
                lens.append(n)
                break
        if not ok:
            fail += 1
    lo = min(lens) if lens else 0
    hi = max(lens) if lens else 0
    print(f"max_len={ml}: fail={fail}/{len(rows)}, range={lo}-{hi}")

print("Qwen2.5-1.5B native context: 32K+; MAX_SEQ_LENGTH=1024 is our 4GB VRAM setting")
