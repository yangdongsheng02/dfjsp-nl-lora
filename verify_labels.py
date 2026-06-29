import json
from pathlib import Path

from transformers import AutoTokenizer

from scheduling_format import build_sft_text
from train_unsloth import MAX_SEQ_LENGTH

tok = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-1.5B", trust_remote_code=True)
row = json.loads(Path("train_data.jsonl").read_text(encoding="utf-8").splitlines()[0])
text = build_sft_text(row["description"], row["optimal_action"], tok, max_length=MAX_SEQ_LENGTH)
ids = tok(text, add_special_tokens=False).input_ids
marker = "### Response:\n"
prefix = text.split(marker, 1)[0] + marker
start = len(tok(prefix, add_special_tokens=False).input_ids)
decoded_from_start = tok.decode(ids[start : start + 20])
print("start index", start, "total", len(ids))
print("decoded from response start:", repr(decoded_from_start))
print("expected:", text.split(marker, 1)[1][:60])
