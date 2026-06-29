import json
from pathlib import Path

from transformers import AutoTokenizer

from scheduling_format import build_inference_prompt, build_sft_text, parse_makespan_completion
from train_unsloth import MAX_SEQ_LENGTH

tok = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-1.5B", trust_remote_code=True)
row = json.loads(Path("train_data.jsonl").read_text(encoding="utf-8").splitlines()[0])
text = build_sft_text(row["description"], row["optimal_action"], tok, max_length=MAX_SEQ_LENGTH)
n = len(tok(text, add_special_tokens=False).input_ids)
prompt = build_inference_prompt(row["description"], tok, max_length=MAX_SEQ_LENGTH)
assert parse_makespan_completion('{"makespan": 56}') == {"makespan": 56}
assert parse_makespan_completion("56}") is None
print("tokens", n)
print("input", text.split("### Input:")[1].split("### Response:")[0].strip())
print("label", text.split("### Response:")[1].strip())
print("OK")
