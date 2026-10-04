"""Combine per-city pair files into chat-format SFT data.  usage: python prep_data.py kumasi accra  -> data/*.jsonl"""
import sys, json, os
SYSTEM = ("You are a navigation assistant for Ghana. Given where the user is (a place name or GPS pin) and where they want "
          "to go, reply with short, spoken, landmark-based directions the way a local would: counted turns, landmarks to "
          "confirm each turn, no distances or coordinates.")
os.makedirs("data", exist_ok=True)
cities = sys.argv[1:]
for split in ("train", "test_area", "test_pair"):
    rows, seen = [], set()
    for c in cities:
        for l in open(f"../{c}_{split}.jsonl"):
            r = json.loads(l); k = (r["input"], r["output"])
            if k in seen: continue
            seen.add(k)
            rows.append(dict(id=r["id"], task=r["task"], city=r["city"],
                             messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["input"]},
                                       {"role": "assistant", "content": r["output"]}]))
    with open(f"data/{split}.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(split, len(rows))
