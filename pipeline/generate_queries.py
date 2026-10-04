"""Scenarios -> natural-language (input, output) pairs with Gemini, then fact-checked against the facts.

usage: python generate_queries.py accra [limit]     (resumable)
Writes <city>_pairs.jsonl (accepted), <city>_rejects.jsonl (dropped, with reason), then
       <city>_train.jsonl / <city>_test_area.jsonl / <city>_test_pair.jsonl.  Needs GEMINI_API_KEY (env or .env).
"""
import sys, os, json, random, time, threading, concurrent.futures as cf
from collections import Counter
import requests
from factcheck import check, index

CITY = sys.argv[1]; LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
KEY = os.environ.get("GEMINI_API_KEY") or dict(l.strip().split("=", 1) for l in open(".env") if "=" in l)["GEMINI_API_KEY"]
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={KEY}"
N = 5

ROUTE_STYLES = [
    "names only (no coordinates), casual, like a Ghanaian texting",
    "start as GPS coordinates (lat, lon) copied exactly from FACTS, destination by name",
    "both start and destination as GPS coordinates copied exactly from FACTS, nothing else",
    "start by name with its coordinates, destination by name; formal and polite",
    "short, rushed, informal but plain English (names only)",
    "mentions only the area/neighbourhood of the start as context, plus the destination name",
    "asks how to get there by car, names only",
]
STYLES = {
    "route": ROUTE_STYLES, "reverse": ROUTE_STYLES,
    "short_hop": ["names only, asks if it is close / how to walk there", "names only, rushed",
                  "start and destination as GPS coordinates copied exactly", "formal, names with coordinates",
                  "informal, names only"],
    "gps_start": ["only the start coordinates (exactly as in FACTS, 5 decimals) + destination name",
                  "'I'm at <lat>, <lon>' sentence + destination name", "pasted location pin text + destination",
                  "formal, with coordinates + destination", "rushed, coordinates + destination, informal"],
    "avoid_landmark": ["traffic/jam at the avoided place, asks for another way", "road blocked at the avoided place",
                       "'I don't want to pass X' phrasing", "polite request for an alternative that skips X",
                       "rushed, informal, avoid X"],
    "avoid_road": ["that road is blocked/bad, want another way", "'avoid <road>' phrasing",
                   "traffic on the road, asks alternative", "polite request not to use the road",
                   "rushed, informal, not that road"],
    "nearby": ["what is around me / near this place", "which landmarks are close to here", "formal: places nearby",
               "rushed, informal: what is around here", "wants to know what to look out for around this point"],
    "connects:link": ["what will I pass between A and B", "how are A and B connected", "main roads/landmarks from A to B",
                      "formal: what is the general way from A to B", "rushed: major landmarks A to B (plain English)"],
    "connects:on_the_way": ["is <question_landmark> on the way from A to B?", "will I pass <question_landmark> going A to B?",
                            "formal: does the route from A to B go by <question_landmark>?",
                            "rushed, informal: is <question_landmark> on the way?", "casual yes/no check on <question_landmark>"],
}
DESC = {
    "route": "Give driving directions from the start to the destination.",
    "reverse": "Give driving directions from the start to the destination (this is the return leg of another trip).",
    "short_hop": "This is a very short trip (a few minutes). Give brief directions in 1-2 sentences.",
    "gps_start": "The user only knows their GPS position (start has no name). First orient them using start.near (the nearest landmark, with its distance/direction), then give the directions.",
    "avoid_landmark": "The user wants a different route that avoids {a}. The FACTS route is the ALTERNATIVE that avoids it: say you are avoiding it, then give these directions. Never route through {a}.",
    "avoid_road": "The user wants a different route that does not use {a}. The FACTS route is the ALTERNATIVE: say you are avoiding it, then give these directions. Never use {a}.",
    "nearby": "Tell the user the landmarks around their position, closest first, using relative wording (right beside, a short walk away). Use FACTS 'nearby' only. 3-5 landmarks in 1-3 sentences.",
    "connects:link": "The user asks how the two places connect. Answer with the key roads and landmarks they pass, in order. NOT turn-by-turn.",
    "connects:on_the_way": "The user asks whether {a} is on the way. Answer yes/no honestly from FACTS (on_route). If yes, say where along the route; if no, say it is a good way off the route, not on the way.",
}
PROMPT = """You create training data for a landmark-based navigation assistant in {city}, Ghana. Locals give directions like: "After the market, take the second left. You'll see the petrol station on your right, that's how you know it's the right turn. Keep going until you reach the church."

FACTS (street-level ground truth from a map; the route, roads, counted turns and sides are fixed):
{facts}

TASK: {task}

Rules for every "output":
- Natural spoken directions in simple, plain English, at most {maxw} words. NO kilometres/metres/distances in numbers, NO coordinates, no compass jargon (a first "head towards ..." is fine).
- Counted turns ("take the second left") must match FACTS exactly. When FACTS say "many small roads", don't count: use a landmark instead ("keep going till you reach ...").
- Reinforce each turn with a landmark the user will SEE there, so they know it is the right turn. Use the landmarks listed in FACTS (keep exact name and side). Where FACTS say NONE, use your own knowledge of {city} to name a well-known landmark that you believe is really at or right next to that junction; if you do not know, guide the user as best you can with the counted turn, the road name and a natural hedge ("if you're not sure, ask someone for ..."). Never contradict FACTS.
- Do not invent road names: use only the roads in FACTS.
- Vary wording between examples. Use simple, plain English that anyone can understand. Do NOT use pidgin, slang or Ghanaian-pidgin words (no "abeg", "dey", "chale", "oale", "you go see", etc.), in either the input or the output.
- "places_used": the road names written in "output", exactly as in FACTS.

Write {n} examples. Style of the "input" (what the user types) for each, in order:
{styles}

Return a JSON array of {n} objects with keys "input", "output", "places_used"."""


def place(p):
    return p["name"] or f"unknown place (GPS {p['lat']}, {p['lon']}; area {p['area']})"


NEAR_WORD = lambda d: "right beside it" if d < 60 else "a short walk away" if d < 250 else "a bit further on"


def render_facts(sc):
    s, e = sc["start"], sc.get("end")
    L = [f"City: {sc['city']}", f"Start: {place(s)} [{s['type']}], area {s['area']}" + (f", GPS pin {s['lat']}, {s['lon']}" if s["name"] is None else f" (GPS {s['lat']}, {s['lon']})")]
    if s.get("near"): L.append(f"Nearest landmark to the start: {s['near']['name']} ({NEAR_WORD(s['near']['dist_m'])})")
    if e: L.append(f"Destination: {e['name']} [{e['type']}], area {e['area']} (GPS {e['lat']}, {e['lon']})")
    if sc.get("nearby"):
        L.append("Nearby landmarks (closest first):"); L += [f"- {n['name']} ({n['type']}), {NEAR_WORD(n['dist_m'])}" for n in sc["nearby"]]
    if sc.get("avoid"): L.append(f"User wants to avoid {sc['avoid']['kind']}: {sc['avoid']['name']} (the route below avoids it)")
    if sc.get("question_landmark"):
        L.append(f"Question landmark: {sc['question_landmark']}; on_route={sc['on_route']}" + ("" if sc["on_route"] else "; it is off the route, a good way away"))
    ORD = {1: "first", 2: "second", 3: "third"}
    for i, st in enumerate(sc.get("steps") or []):
        road = st["road"] + (" (generic, unnamed: do not name it)" if st["generic"] else "")
        lms = ", ".join(f"{x['name']} (on your {x['side']})" for x in st.get("turn_lms", []))
        lm = f" Landmarks you will see at this turn: {lms}." if lms else " Landmarks at this turn: NONE."
        if i == 0:
            L.append(f"1. Start by heading {st['heading']} on {road}" + (f", passing about {st['side_roads_passed']} small side roads before the first turn." if st["side_roads_passed"] > 1 else "."))
            continue
        turn = st["turn"]; side = "left" if "left" in turn else "right" if "right" in turn else None
        if side:
            ordn = f"the {ORD[st['turn_ordinal']]} {side} turn" if st.get("turn_ordinal") else f"a {side} turn after many small roads (do NOT count them)"
            L.append(f"{i+1}. TURN {side.upper()} = {ordn}, onto {road}.{lm}")
        else:
            L.append(f"{i+1}. KEEP STRAIGHT onto {road}.{lm}")
    if sc.get("steps"):
        near = f" Landmark beside the destination: {sc['dest_near'][0]['name']}." if sc.get("dest_near") else ""
        L.append((f"Destination is on your {sc['dest_side']}." if sc["dest_side"] != "ahead" else "Destination is straight ahead.") + near)
    return "\n".join(L)


def gen(sc):
    key = f"connects:{sc['variant']}" if sc["task"] == "connects" else sc["task"]
    rnd = random.Random(sc["id"])
    styles = rnd.sample(STYLES.get(key, STYLES["route"]), min(N, len(STYLES.get(key, STYLES["route"]))))
    a = (sc.get("avoid") or {}).get("name") or sc.get("question_landmark", "")
    p = PROMPT.format(city=sc["city"], maxw=int(0.75 * __import__("factcheck").MAX_WORDS[sc["task"]]), facts=render_facts(sc),
                      task=DESC.get(key, DESC["route"]).format(a=a), n=len(styles),
                      styles="\n".join(f"  {i+1}. {s}" for i, s in enumerate(styles)))
    body = {"contents": [{"parts": [{"text": p}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 1.0,
                                 "thinkingConfig": ({"thinkingBudget": 0} if MODEL.startswith("gemini-2") else {"thinkingLevel": "minimal"})}}
    err = None
    for attempt in range(5):
        try:
            r = requests.post(URL, json=body, timeout=120)
            if r.status_code in (429, 503): time.sleep(4 * (attempt + 1)); continue
            if not r.ok: raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            arr = json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])
            ix, ok, bad, seen = index(sc), [], [], set()
            for x in arr:
                if not isinstance(x, dict): continue
                why = check(sc, x, ix)
                if not why and x["input"].strip().lower() in seen: why = "duplicate"
                row = dict(id=sc["id"], task=sc["task"], split=sc["split"], city=sc["city"], input=x.get("input"),
                           output=x.get("output"), start=sc["start"]["name"], end=(sc["end"] or {}).get("name"))
                (bad if why else ok).append(dict(row, reason=why) if why else row)
                seen.add(str(x.get("input", "")).strip().lower())
            return ok, bad
        except Exception as e:
            err = e
    print("fail", sc["id"], err, file=sys.stderr); return [], []


def export():
    rows = [json.loads(l) for l in open(f"{CITY}_pairs.jsonl")]
    for split in ("train", "test_area", "test_pair"):
        with open(f"{CITY}_{split}.jsonl", "w") as f:
            for r in rows:
                if r["split"] == split: f.write(json.dumps(dict(input=r["input"], output=r["output"], task=r["task"], city=r["city"], id=r["id"]), ensure_ascii=False) + "\n")
    print({s: sum(r["split"] == s for r in rows) for s in ("train", "test_area", "test_pair")})


if __name__ == "__main__":
    pf = f"{CITY}_pairs.jsonl"
    done = {json.loads(l)["id"] for l in open(pf)} | {json.loads(l)["id"] for l in open(f"{CITY}_rejects.jsonl")} \
        if os.path.exists(pf) and os.path.exists(f"{CITY}_rejects.jsonl") else set()
    todo = [s for s in (json.loads(l) for l in open(f"{CITY}_scenarios.jsonl")) if s["id"] not in done][:LIMIT]
    print(len(todo), "scenarios to do"); acc, rej = Counter(), Counter()
    with open(pf, "a") as fo, open(f"{CITY}_rejects.jsonl", "a") as fr, cf.ThreadPoolExecutor(int(os.environ.get("WORKERS", 32))) as ex:
        for k, (ok, bad) in enumerate(ex.map(gen, todo)):
            for r in ok: fo.write(json.dumps(r, ensure_ascii=False) + "\n"); acc[r["task"]] += 1
            for r in bad: fr.write(json.dumps(r, ensure_ascii=False) + "\n"); rej[r["reason"]] += 1
            fo.flush(); fr.flush()
            if k % 50 == 0: print(k, "/", len(todo), "accepted", sum(acc.values()), "rejected", dict(rej), flush=True)
    print("accepted by task:", dict(acc)); print("rejected by reason:", dict(rej)); export()
