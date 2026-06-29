import json
from pathlib import Path
from transformers import AutoTokenizer
from scheduling_format import build_sft_text, build_prompt_text, label_start_index, format_schedule_nl, parse_schedule_completion

tok = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
row = json.loads(Path("train_data.jsonl").read_text(encoding="utf-8").splitlines()[0])
text = build_sft_text(row["description"], row["optimal_action"], tok, max_length=1536)
ids = tok(text, add_special_tokens=False).input_ids
start = label_start_index(text, ids, tok)
label = tok.decode(ids[start:], skip_special_tokens=True)
gold = format_schedule_nl(row["optimal_action"])
print("match", label.strip() == gold.strip())
print("label head:", label[:200])
prompt = build_prompt_text(row["description"], tok, max_length=1536)
print("prompt tail:", repr(prompt[-120:]))
