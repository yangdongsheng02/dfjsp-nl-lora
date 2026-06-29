import torch
from unsloth import FastLanguageModel
from train_unsloth import MODEL_NAME, MAX_SEQ_LENGTH

model, tok = FastLanguageModel.from_pretrained(MODEL_NAME, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=True, dtype=torch.float16)
FastLanguageModel.for_inference(model)
msgs = [
    {"role":"system","content":"你是助手，用中文简洁回答。"},
    {"role":"user","content":"车间有4台机器，当前有2个急单插入，请用一句话说明调度要考虑什么？"},
]
p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ids = tok(p, add_special_tokens=False)["input_ids"]
with torch.no_grad():
    out = model.generate(torch.tensor([ids],device=model.device), attention_mask=torch.ones(1,len(ids),device=model.device), max_new_tokens=80, do_sample=False, eos_token_id=tok.eos_token_id)
print(tok.decode(out[0,len(ids):], skip_special_tokens=True))
