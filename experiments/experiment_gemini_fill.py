"""Experiment: skeleton route + "fill gaps from your own Ghana knowledge", then verify Gemini's additions against OSM.

usage: python experiment_gemini_fill.py kumasi 25
"""
import sys, os, json, random, time
import numpy as np, requests
from rapidfuzz import fuzz
from world import load_world, bearing, hav
import concurrent.futures as cf

CITY, N = sys.argv[1], int(sys.argv[2])
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
KEY = os.environ.get("GEMINI_API_KEY") or dict(l.strip().split("=", 1) for l in open(".env") if "=" in l)["GEMINI_API_KEY"]
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={KEY}"
ORD = {1: "first", 2: "second", 3: "third", 4: "fourth"}
w = load_world(CITY); G, co = w.G, w.coord
imp = np.array([l["imp"] for l in w.poi])


def side_roads(path, i):
    n = path[i]; prev, nxt = path[i - 1], path[i + 1]; b = bearing(*co[prev], *co[n]); out = []
    for m in set(G.successors(n)) | set(G.predecessors(n)):
        if m in (prev, nxt): continue
        d = (bearing(*co[n], *co[m]) - b + 540) % 360 - 180
        if 25 < d < 155: out.append("right")
        elif -155 < d < -25: out.append("left")
    return out


def landmark_at(node, path_prev, r=100):
    """best graph landmark within r m of a junction, with the side of the road it is on"""
    la, lo = co[node]; d = w._d(w.poi_xy, la, lo); ix = [i for i in np.argsort(d)[:10] if d[i] <= r]
    if not ix: return None
    i = max(ix, key=lambda k: (imp[k], -d[k])); l = w.poi[i]
    b_in = bearing(*co[path_prev], la, lo)
    rel = (bearing(la, lo, l["lat"], l["lon"]) - b_in + 540) % 360 - 180
    return dict(name=l["name"], side="right" if 15 < rel < 165 else "left" if -165 < rel < -15 else "ahead")


def skeleton(path, st):
    idx = {n: i for i, n in enumerate(path)}; L = []; turns = []
    L.append(f"1. Start by heading {st[0]['heading']} on {st[0]['road']}.")
    for k in range(1, len(st)):
        t = st[k]; turn = t["turn"]; side = "left" if "left" in turn else "right" if "right" in turn else None
        p0, i0 = idx[st[k - 1]["start_node"]], idx[t["start_node"]]
        passed = [x for j in range(p0 + 1, i0 + 1) if j < len(path) - 1 for x in side_roads(path, j)]
        lm = landmark_at(t["start_node"], path[max(i0 - 1, 0)])
        if side:
            o = passed.count(side) + 1   # +1: the turn itself
            what = f"{ORD.get(o, str(o)+'th')} {side} turn" if o <= 3 else f"{side} turn (many small roads before it, so do NOT count them)"
            L.append(f"{k+1}. TURN {side.upper()} ({what}) onto {t['road']}." + (f" Small roads skipped on the way: {passed.count(side)} on that side, {len(passed)-passed.count(side)} on the other." if passed else ""))
        else:
            L.append(f"{k+1}. KEEP STRAIGHT onto {t['road']}, passing {len(passed)} small side roads.")
        L[-1] += (f" GRAPH landmark at this junction: {lm['name']} (on your {lm['side']})." if lm else " GRAPH landmark at this junction: NONE.")
        turns.append((k + 1, t["start_node"]))
    return "\n".join(L), turns


PROMPT = """You create training data for a landmark-based navigation assistant in {city}, Ghana. Locals give directions like: "After the market, take the second left. You'll see the petrol station on your right, that's how you know it's the right turn. Then keep going until you reach the church."

SKELETON of a real route from {a} to {b} (street-level facts from a map; ground truth):
{sk}
Destination is on your {side}.

Write {n} different examples (user request + assistant answer). The user request styles: 1) names only casual, 2) start given as GPS pin "{lat}, {lon}" + destination name, 3) rushed light pidgin, 4) polite, 5) just destination with "I'm near {a}".
Rules for each answer:
- Natural spoken directions, max 80 words. NO kilometres/metres, NO coordinates, NO compass-direction jargon beyond what a local would say.
- Counted turns ("take the second left") must match the skeleton exactly. Never count more than three; follow the skeleton's wording on that.
- Reinforce each turn with a landmark the user will see there so they know it's the right turn. Use the GRAPH landmark when given (keep its exact name and side).
- Where the skeleton says NONE you MAY use your own knowledge of {city} to add a real, well-known landmark that you are confident is actually at or right next to that junction; if you are not confident, add nothing and rely on the counted turn. List every landmark you added from your own knowledge in "added" as {{"step": <number>, "name": "<exact name>"}}.
- Do not invent road names; use only the ones in the skeleton.
Return a JSON array of {n} objects with keys "input", "output", "added"."""


def call(prompt):
    body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {
        "responseMimeType": "application/json", "temperature": 1.0,
        "thinkingConfig": {"thinkingBudget": 0} if MODEL.startswith("gemini-2") else {"thinkingLevel": "minimal"}}}
    for a in range(4):
        try:
            r = requests.post(URL, json=body, timeout=120)
            if r.status_code in (429, 503): time.sleep(4 * (a + 1)); continue
            if not r.ok: raise RuntimeError(r.text[:150])
            return json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])
        except Exception as e: err = e
    return []


rnd = random.Random(11)
sc = [json.loads(l) for l in open(f"{CITY}_scenarios.jsonl") if '"task": "route"' in l]
by = {l["name"]: l for l in w.LM}
jobs = []
for s in rnd.sample(sc, 400):
    a, b = by.get(s["start"]["name"]), by.get(s["end"]["name"])
    if not a or not b or s["route_km"] > 6: continue
    path = w.route(a["node"], b["node"]); st = w.steps(path, exclude={a["name"], b["name"]})
    if not 3 <= len(st) <= 5: continue
    sk, turns = skeleton(path, st)
    p = PROMPT.format(city=CITY.title(), a=a["name"], b=b["name"], sk=sk, side=w.dest_side(path, b), n=5,
                      lat=a["lat"], lon=a["lon"])
    jobs.append((a, b, sk, turns, p))
    if len(jobs) == N: break
with cf.ThreadPoolExecutor(16) as ex: res = list(ex.map(lambda j: call(j[4]), jobs))

stat = dict(added=0, verified_near=0, real_but_far=0, not_in_osm=0, answers=0, turns_no_graph_lm=0)
rows = []
for (a, b, sk, turns, p), out in zip(jobs, res):
    stat["turns_no_graph_lm"] += sk.count("NONE")
    for x in out:
        stat["answers"] += 1
        for ad in x.get("added") or []:
            stat["added"] += 1; nm = str(ad.get("name", "")); step = ad.get("step")
            node = dict(turns).get(step)
            cands = [(fuzz.token_set_ratio(nm.lower(), l["name"].lower()), l) for l in w.LM]
            sc_, l = max(cands, key=lambda c: c[0])
            if sc_ < 90: stat["not_in_osm"] += 1; verdict = "NOT IN OSM"
            else:
                dist = hav(*co[node], l["lat"], l["lon"]) if node else 9e9
                if dist <= 150: stat["verified_near"] += 1; verdict = f"OK ({int(dist)} m)"
                else: stat["real_but_far"] += 1; verdict = f"REAL BUT FAR ({int(dist)} m)"
            rows.append((a["name"], b["name"], step, nm, verdict))
print(MODEL, stat)
print("\nsample of Gemini-added landmarks and verdicts:")
for r in rows[:25]: print(" ", r)
print("\n=== one full example ===")
a, b, sk, turns, p = jobs[0]; print(sk); print(json.dumps(res[0][:3], indent=1, ensure_ascii=False))
