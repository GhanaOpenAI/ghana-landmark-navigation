"""SFT (assistant-only loss) with plain HF Trainer.  usage: python train.py --model Qwen/Qwen3.5-0.8B --out runs/q08 [--full]"""
import argparse, json, random, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
p = argparse.ArgumentParser()
p.add_argument("--model", required=True); p.add_argument("--out", required=True)
p.add_argument("--full", action="store_true", help="full fine-tune instead of LoRA")
p.add_argument("--epochs", type=float, default=2); p.add_argument("--lr", type=float)
p.add_argument("--bs", type=int, default=4); p.add_argument("--accum", type=int, default=4); p.add_argument("--max_len", type=int, default=768)
p.add_argument("--limit", type=int, default=0); p.add_argument("--r", type=int, default=64)
a = p.parse_args()
tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token


def render(msgs, gen):
    kw = dict(tokenize=False, add_generation_prompt=gen)
    try: return tok.apply_chat_template(msgs, enable_thinking=False, **kw)
    except TypeError: return tok.apply_chat_template(msgs, **kw)


def encode(r):
    m = r["messages"]; prompt = render(m[:-1], True); full = prompt + m[-1]["content"] + (tok.eos_token or "")
    pi = tok(prompt, add_special_tokens=False)["input_ids"]; fi = tok(full, add_special_tokens=False)["input_ids"][:a.max_len]
    return dict(input_ids=fi, labels=[-100] * len(pi) + fi[len(pi):])


rows = [json.loads(l) for l in open("data/train.jsonl")]
random.Random(0).shuffle(rows)
if a.limit: rows = rows[:a.limit]
ds = [encode(r) for r in rows]
print("train examples", len(ds), "| avg tokens", sum(len(d["input_ids"]) for d in ds) / len(ds))


def collate(b):
    L = max(len(x["input_ids"]) for x in b)
    ids = torch.full((len(b), L), tok.pad_token_id); lab = torch.full((len(b), L), -100); att = torch.zeros((len(b), L), dtype=torch.long)
    for i, x in enumerate(b):
        n = len(x["input_ids"]); ids[i, :n] = torch.tensor(x["input_ids"]); lab[i, :n] = torch.tensor(x["labels"]); att[i, :n] = 1
    return dict(input_ids=ids, labels=lab, attention_mask=att)


torch.backends.cuda.matmul.allow_tf32 = True
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa").to("cuda")
if not a.full:
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(r=a.r, lora_alpha=a.r * 2, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
lr = a.lr or (2e-5 if a.full else 2e-4)
args = TrainingArguments(output_dir=a.out, per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.accum, num_train_epochs=a.epochs, learning_rate=lr,
                         lr_scheduler_type="cosine", warmup_steps=max(1, int(0.03 * len(ds) / (a.bs * a.accum) * a.epochs)), bf16=True, logging_steps=25, save_strategy="no",
                         report_to=[], remove_unused_columns=False, dataloader_num_workers=2, optim="adamw_torch_fused")
Trainer(model=model, args=args, train_dataset=ds, data_collator=collate).train()
model.save_pretrained(a.out); tok.save_pretrained(a.out)
json.dump(vars(a), open(f"{a.out}/train_args.json", "w"))
print("saved", a.out)
