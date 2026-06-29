import json
from pathlib import Path
from scheduling_format import build_prompt_text, parse_schedule_completion, generate_schedule, prompt_input_ids
from train_unsloth import MAX_SEQ_LENGTH, MODEL_NAME
from unsloth import FastLanguageModel
import torch

row = json.loads(Path("test_data.jsonl").read_text(encoding="utf-8").splitlines()[0])
model, tok = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=True, dtype=torch.float16,
)
FastLanguageModel.for_inference(model)
prompt = build_prompt_text(row["description"], tok, max_length=MAX_SEQ_LENGTH)
ids = prompt_input_ids(row["description"], tok, max_length=MAX_SEQ_LENGTH)
print(f"prompt_tokens={len(ids)} gen_budget={MAX_SEQ_LENGTH-len(ids)}")
print(f"prompt_tail:\n{prompt[-300:]}\n")
raw = generate_schedule(model, tok, row["description"], max_length=MAX_SEQ_LENGTH, decode="fast")
print(f"raw[:800]:\n{raw[:800]}\n")
pred = parse_schedule_completion(raw)
print(f"parse={pred is not None} ops={len(pred['schedule']) if pred else 0} ms={pred.get('makespan') if pred else None}")
