import json
from pathlib import Path
from transformers import AutoTokenizer
from scheduling_format import build_sft_text, build_prompt_text

tok = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
row = json.loads(Path("train_data.jsonl").read_text(encoding="utf-8").splitlines()[0])
text = build_sft_text(row["description"], row["optimal_action"], tok, max_length=1536)
prompt = build_prompt_text(row["description"], tok, max_length=1536)
print("PROMPT IDS", len(tok(prompt, add_special_tokens=False).input_ids))
print("FULL IDS", len(tok(text, add_special_tokens=False).input_ids))
print("---prompt tail---")
print(repr(prompt[-150:]))
print("---full around boundary---")
p = len(prompt)
print(repr(text[max(0,p-50):p+80]))
