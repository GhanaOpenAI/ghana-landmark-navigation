"""Generate on held-out splits and score against the structured facts.
usage: python eval.py --model Qwen/Qwen3.5-0.8B --adapter runs/q08 --n 200 --out results/q08.json"""
import argparse, json, re, random, sys, torch
from collections import defaultdict
sys.path.insert(0, "."); sys.path.insert(0, "../pipeline"); from factcheck import check, index, PLACE_RE, norm
from transformers import AutoTokenizer, AutoModelForCausalLM
p = argparse.ArgumentParser()
p.add_argument("--model", required=True); p.add_argument("--adapter"); p.add_argument("--n", type=int, default=200)
p.add_argument("--out", required=True); p.add_argument("--bs", type=int, default=32)
a = p.parse_args()
tok = AutoTokenizer.from_pretrained(a.adapter or a.model, trust_remote_code=True, padding_side="left")
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
if a.adapter:
    from peft import PeftModel
    try: model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    except Exception: model = AutoModelForCausalLM.from_pretrained(a.adapter, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
model.eval()
SC = {}
for c in ("kumasi", "accra"):
    try:
        for l in open(f"scenarios/{c}_scenarios.jsonl"): d = json.loads(l); SC[d["id"]] = d
    except FileNotFoundError: pass


def prompt(msgs):
    kw = dict(tokenize=False, add_generation_prompt=True)
    try: return tok.apply_chat_template(msgs, enable_thinking=False, **kw)
    except TypeError: return tok.apply_chat_template(msgs, **kw)


def road_set(sc): return {norm(s["road"]) for s in sc.get("steps") or [] if not s.get("generic")}


res, samples = {}, []
for split in ("test_pair", "test_area"):
    rows = [json.loads(l) for l in open(f"data/{split}.jsonl")]; random.Random(1).shuffle(rows); rows = rows[:a.n]
    stat = defaultdict(float); n = 0; bytask = defaultdict(lambda: [0, 0])
    for i in range(0, len(rows), a.bs):
        b = rows[i:i + a.bs]
        enc = tok([prompt(r["messages"][:-1]) for r in b], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad(): out = model.generate(**enc, max_new_tokens=220, do_sample=False, pad_token_id=tok.pad_token_id)
        for r, o in zip(b, out):
            txt = tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip(); sc = SC.get(r["id"]); n += 1
            if not sc: continue
            why = check(sc, dict(input=r["messages"][1]["content"], output=txt))
            stat["pass"] += why is None; bytask[r["task"]][0] += why is None; bytask[r["task"]][1] += 1
            if why: stat["fail_" + why] += 1
            routes = road_set(sc); said = {norm(m.group(1)) for m in PLACE_RE.finditer(txt)}
            ok = {s for s in said if any(s in x or x in s for x in routes)}
            stat["road_precision"] += (len(ok) / len(said)) if said else 1.0
            stat["road_recall"] += (len(ok) / len(routes)) if routes else 1.0
            stat["names_dest"] += bool(sc.get("end")) and norm(sc["end"]["name"]) in norm(txt) if sc.get("end") else 0
            if len(samples) < 40 and i == 0: samples.append(dict(split=split, task=r["task"], input=r["messages"][1]["content"], ref=r["messages"][2]["content"], pred=txt, verdict=why))
    res[split] = {k: round(v / n, 3) for k, v in stat.items()}; res[split]["n"] = n
    res[split]["by_task_pass"] = {t: round(x[0] / x[1], 2) for t, x in bytask.items()}
    print(split, json.dumps(res[split]))
import os; os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(dict(model=a.model, adapter=a.adapter, results=res, samples=samples), open(a.out, "w"), indent=1, ensure_ascii=False)
