"""Sample structured scenarios from the task registry, with a held-out split.

usage: python build_scenarios.py accra 4000   ->  <city>_scenarios.jsonl, <city>_landmarks.csv

split: scenarios touching a held-out AREA (10% of areas) -> test_area  (tests generalising to unseen places)
       else 5% of landmark PAIRS                         -> test_pair  (unseen route between seen places)
       else                                              -> train
"""
import sys, json, csv, random, hashlib
from collections import Counter
from world import load_world
from tasks import TASKS

CITY, N = sys.argv[1], int(sys.argv[2])
rnd = random.Random(7)
w = load_world(CITY)
print(CITY, "graph", w.G.number_of_nodes(), "nodes;", len(w.poi), "landmarks,", len(w.areas), "areas")

with open(f"{CITY}_landmarks.csv", "w", newline="") as f:
    cw = csv.DictWriter(f, fieldnames=["id", "name", "lat", "lon", "cat", "imp", "area", "node", "suburb", "source"]); cw.writeheader()
    cw.writerows(w.LM)


def h(s, mod): return int(hashlib.md5(s.encode()).hexdigest(), 16) % mod


def split_of(sc):
    areas = [(sc[k] or {}).get("area") or "" for k in ("start", "end")]
    if any(a and h("area:" + a, 10) == 0 for a in areas): return "test_area"
    if h("|".join(sorted(sc["pair"])), 20) == 0: return "test_pair"
    return "train"


import time; T0 = time.time()
names = list(TASKS); weights = [TASKS[n][1] for n in names]
out, seen, tries = [], set(), 0
while len(out) < N and tries < N * 60:
    tries += 1
    if tries % 2000 == 0: print(f"  {len(out)}/{N} scenarios, {tries} tries, {time.time()-T0:.0f}s elapsed, ETA ~{(time.time()-T0)/max(len(out),1)*(N-len(out)):.0f}s", flush=True)
    for sc in TASKS[rnd.choices(names, weights)[0]][0](w, rnd):
        key = (sc["task"], tuple(sc["pair"]), sc.get("avoid", {}).get("name"), sc.get("question_landmark"),
               sc["start"].get("lat") if sc["start"].get("name") is None else None)
        if key in seen: continue
        seen.add(key); out.append(sc)
for i, sc in enumerate(out):
    sc["id"] = f"{CITY}_{i:05d}"; sc["split"] = split_of(sc)
with open(f"{CITY}_scenarios.jsonl", "w") as f:
    for sc in out: f.write(json.dumps(sc, ensure_ascii=False) + "\n")
print("scenarios", len(out), "| tries", tries)
print("by task :", dict(Counter(s["task"] for s in out)))
print("by split:", dict(Counter(s["split"] for s in out)))
steps = [len(s["steps"]) for s in out if s.get("steps")]
gen = sum(st["generic"] for s in out for st in s.get("steps", []))
print(f"steps/route avg {sum(steps)/len(steps):.1f}, max {max(steps)}; generic-road steps {gen}/{sum(steps)}")
