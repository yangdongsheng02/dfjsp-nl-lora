import json
from pathlib import Path

from transformers import AutoTokenizer

from scheduling_format import build_sft_text, label_start_index
from train_unsloth import MAX_SEQ_LENGTH

tok = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-1.5B", trust_remote_code=True)
rows = [json.loads(l) for l in Path("train_data.jsonl").read_text(encoding="utf-8").splitlines()]
lengths = []
for r in rows:
    t = build_sft_text(r["description"], r["optimal_action"], tok, max_length=MAX_SEQ_LENGTH)
    n = len(tok(t, add_special_tokens=False).input_ids)
    lengths.append(n)
    label = t.split("### Response:", 1)[1].strip()
    assert label.startswith("{"), label[:80]
    ids = tok(t, add_special_tokens=False).input_ids
    start = label_start_index(t, ids, tok)
    assert tok.decode(ids[start:]).startswith('{"makespan":'), tok.decode(ids[start:])[:40]
print("tokens min/max/mean:", min(lengths), max(lengths), round(sum(lengths) / len(lengths), 1))
print("all <=512:", all(x <= 512 for x in lengths))
print("sample label:", t.split("### Response:", 1)[1].strip())
