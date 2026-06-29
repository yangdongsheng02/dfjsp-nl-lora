import json
from pathlib import Path
from transformers import AutoTokenizer
from scheduling_format import INSTRUCTION, format_schedule_nl, parse_schedule_completion, RESPONSE_MARKER

tok = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
print("chat_template:", bool(tok.chat_template))
row = json.loads(Path("test_data.jsonl").read_text(encoding="utf-8").splitlines()[0])

# Alpaca style
alpaca = f"### Instruction:\n{INSTRUCTION}\n\n### Input:\n{row['description'][:200]}...\n\n{RESPONSE_MARKER}"
# Chat style
msgs = [{"role":"system","content":INSTRUCTION},{"role":"user","content":row["description"]}]
chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print("=== alpaca tail ===", repr(alpaca[-80:]))
print("=== chat tail ===", repr(chat[-200:]))
