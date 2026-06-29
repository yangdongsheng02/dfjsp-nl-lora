import json
from pathlib import Path
from transformers import AutoTokenizer
from scheduling_format import (
    INSTRUCTION, build_prompt_text, build_sft_text, format_schedule_nl,
    parse_schedule_completion, prompt_input_ids, generate_schedule, NL_OP_RE
)
from train_unsloth import MAX_SEQ_LENGTH, MODEL_NAME

tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
row = json.loads(Path("test_data.jsonl").read_text(encoding="utf-8").splitlines()[0])

# 1) parser sanity
label = format_schedule_nl(row["optimal_action"])
parsed = parse_schedule_completion(label)
print("=== parser on gold label ===")
print(f"ops={len(parsed['schedule'])} ms={parsed['makespan']} complete={parsed['_complete']}")

# 2) prompt stats
prompt = build_prompt_text(row["description"], tok, max_length=MAX_SEQ_LENGTH)
ids = prompt_input_ids(row["description"], tok, max_length=MAX_SEQ_LENGTH)
print(f"\n=== prompt ===")
print(f"prompt_tokens={len(ids)} gen_budget={MAX_SEQ_LENGTH-len(ids)}")
print(f"tail80: {repr(prompt[-80:])}")

# 3) load model quick test
from unsloth import FastLanguageModel
import torch
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=True, dtype=torch.float16,
)
FastLanguageModel.for_inference(model)

raw = generate_schedule(model, tokenizer, row["description"], max_length=MAX_SEQ_LENGTH, decode="fast")
print(f"\n=== fast decode (first 500 chars) ===")
print(repr(raw[:500]))
pred = parse_schedule_completion(raw)
print(f"parse={pred is not None} ops={len(pred['schedule']) if pred else 0}")

# 4) try chat template
messages = [
    {"role": "system", "content": INSTRUCTION},
    {"role": "user", "content": row["description"]},
]
chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
chat_ids = tokenizer(chat_prompt, add_special_tokens=False)["input_ids"]
print(f"\n=== chat template ===")
print(f"chat_tokens={len(chat_ids)} budget={MAX_SEQ_LENGTH-len(chat_ids)}")
inputs = {"input_ids": torch.tensor([chat_ids], device=model.device), "attention_mask": torch.ones(1,len(chat_ids),device=model.device)}
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=min(200, MAX_SEQ_LENGTH-len(chat_ids)), do_sample=False)
chat_raw = tokenizer.decode(out[0,len(chat_ids):], skip_special_tokens=True)
print(repr(chat_raw[:500]))
print(f"chat parse={parse_schedule_completion(chat_raw) is not None}")
