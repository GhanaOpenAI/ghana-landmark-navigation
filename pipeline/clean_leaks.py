"""Remove every scenario whose generated pairs contain a leaked-style or location-less input, so it regenerates."""
import json, re, sys
CITY = sys.argv[1]
LEAK = re.compile(r"^(formal|rushed|casual|polite|informal|short)\s*:|yes/no check|\bnames only\b|\b[AB] (and|to) [AB]\b|<[^>]+>", re.I)
S = {json.loads(l)["id"]: json.loads(l) for l in open(f"{CITY}_scenarios.jsonl")}
pairs = [json.loads(l) for l in open(f"{CITY}_pairs.jsonl")]
bad = set()
for p in pairs:
    sc = S.get(p["id"]); inp = p["input"]
    if LEAK.search(inp): bad.add(p["id"])
    elif sc and sc["task"] == "nearby":
        nm = (sc["start"].get("name") or "")
        if not ((nm and nm.lower() in inp.lower()) or str(sc["start"]["lat"])[:6] in inp): bad.add(p["id"])
print(CITY, "scenarios to regenerate:", len(bad), "| pairs removed:", sum(p["id"] in bad for p in pairs))
for fn in (f"{CITY}_pairs.jsonl", f"{CITY}_rejects.jsonl"):
    rows = [l for l in open(fn) if json.loads(l)["id"] not in bad]
    open(fn, "w").writelines(rows); print(" ", fn, "now", len(rows))
