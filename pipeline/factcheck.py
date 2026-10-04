"""Validate an LLM-written (input, output) pair against its structured scenario. check() -> reason or None.

Catches: invented/undeclared place names, distances that don't match the route, wrong coordinates,
left/right or compass words the route doesn't contain, over-long answers, malformed rows.
"""
import re
from rapidfuzz import fuzz

MAX_WORDS = dict(route=110, reverse=110, short_hop=60, avoid_landmark=115, avoid_road=115, gps_start=115,
                 nearby=70, connects=90)   # hard cap; the prompt asks for ~75% of this
PLACE_WORDS = ("Road|Street|Avenue|Highway|Junction|Roundabout|Circle|Interchange|Market|Hospital|School|Station|Park|"
               "Lane|Close|Church|Mosque|Hotel|Bank|Terminal|Stadium|Mall|Cathedral|University|College|Clinic|Square|Boulevard|Bypass|Drive|Way|Crescent|Link|Estate|Gate|Plaza|Centre|Center|Clinic|Chapel|Junction")
ROAD_WORDS = "Road|Street|Avenue|Highway|Bypass|Boulevard|Drive|Crescent|Lane|Close|Interchange|Way|Link"
PLACE_RE = re.compile(rf"((?:[A-Z][\w'’.&-]*\s+){{0,5}}(?:{ROAD_WORDS})(?:\s+[A-Z][\w'’.&-]*){{0,3}})\b")
SIDE_RE = re.compile(r"\b(?:on your|to your|to the|your|turn|turns|take a|make a|bear|keep|veer|slight|sharp|a|then|and)\s+(left|right)\b", re.I)
DIST_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(km|kilomet\w*|metres?|meters?|m)\b", re.I)
COORD_RE = re.compile(r"-?\d{1,2}\.\d{3,6}")
ORD_RE = re.compile(r"\b(first|second|third|1st|2nd|3rd)\s+(?:slight\s+|sharp\s+)?(left|right)\b", re.I)
DIR_RE = re.compile(r"\b(north[- ]?east|north[- ]?west|south[- ]?east|south[- ]?west|north|south|east|west)\b", re.I)


def norm(s): return re.sub(r"[^a-z0-9 ]", "", s.lower().replace("-", " ")).strip()
def ndir(s): return re.sub(r"[ -]", "", s.lower())


def index(sc):
    """everything the facts allow: names, distances (m), directions, turns, coordinates"""
    names, dists, dirs, turns, coords, ords = set(), set(), set(), set(), [], set()

    def walk(o, key=""):
        if isinstance(o, dict):
            if o.get("lat") is not None: coords.extend([o["lat"], o["lon"]])
            for k, v in o.items():
                if k in ("name", "area") and isinstance(v, str): names.add(norm(v))
                elif k == "road" and isinstance(v, str) and not o.get("generic"): names.add(norm(v))
                elif k in ("heading", "direction", "bearing") and isinstance(v, str): dirs.add(ndir(v))
                elif k == "turn": turns.add(v)
                elif k in ("dest_side", "side"): turns.add(v)
                elif (k.endswith("dist_m") or k.endswith("_m")) and isinstance(v, (int, float)): dists.add(float(v))
                elif k.endswith("_km") and isinstance(v, (int, float)): dists.add(v * 1000.0)
                elif k in ("landmarks_at_end",): names.update(norm(x) for x in v)
                elif k in ("question_landmark", "on_road") and isinstance(v, str): names.add(norm(v))
                walk(v, k)
        elif isinstance(o, list):
            for x in o: walk(x, key)
    walk(sc)
    st = [s["dist_m"] for s in sc.get("steps", [])]
    for i in range(len(st)):                       # contiguous stretches ("about 2 km along ...")
        for j in range(i + 1, len(st) + 1): dists.add(float(sum(st[i:j])))
    for s in sc.get("steps") or []:                 # numbered turns: ("second", "left") etc.
        tn = s.get("turn") or ""; side = "left" if "left" in tn else "right" if "right" in tn else None
        if side and s.get("turn_ordinal"): ords.add(({1: "first", 2: "second", 3: "third"}[s["turn_ordinal"]], side))
    ords |= {({"1st": "first", "2nd": "second", "3rd": "third"}.get(a, a), b) for a, b in list(ords)}
    names.discard("")
    return names, dists, dirs, turns, coords, ords


def _metres(num, unit):
    v = float(num.replace(",", "."))
    return v * 1000 if unit.lower().startswith("k") else v


def check(sc, row, idx=None):
    names, dists, dirs, turns, coords, ords = idx or index(sc)
    inp, out = str(row.get("input", "")).strip(), str(row.get("output", "")).strip()
    if not inp or not out: return "empty"
    if len(inp.split()) > 70: return "input_long"
    if len(out.split()) > MAX_WORDS.get(sc["task"], 85): return "output_long"
    blob = "|".join(names)
    for p in row.get("places_used") or []:         # declared ROAD names must exist in the facts (landmarks are free)
        n = norm(str(p))
        if n and str(p).split()[-1].lower() in ROAD_WORDS.lower().split("|") and n not in names and not any(fuzz.ratio(n, x) >= 88 for x in names) and n not in blob:
            return "road_declared"
    keyw = set(ROAD_WORDS.lower().split("|"))
    name_toks = [set(x.split()) for x in names]
    for m in PLACE_RE.finditer(out):               # undeclared place-looking phrases must exist too
        words = m.group(1).split()
        ok = False
        for i in range(len(words)):
            for j in range(len(words), i, -1):
                span = norm(" ".join(words[i:j])); toks = set(span.split())
                if len(toks) < 2 or toks <= keyw: continue
                if span in blob or any(toks <= nt for nt in name_toks): ok = True; break
            if ok: break
        if not ok and len(words) >= 2 and not all(norm(x) in keyw for x in words): return "road_undeclared"
    if DIST_RE.search(out): return "has_distance"          # locals don't say km/metres: keep answers relative
    if COORD_RE.search(out): return "coord_in_output"
    for c in COORD_RE.findall(inp):                         # coordinates in the INPUT must be the scenario's
        if not any(abs(float(c) - x) <= 0.0006 for x in coords): return "coord"
    for m in ORD_RE.finditer(out):                          # "second left" must be a real numbered turn
        if (m.group(1).lower(), m.group(2).lower()) not in ords: return "ordinal"
    if re.search(r"\b(fourth|fifth|sixth|seventh|4th|5th|6th)\s+(left|right|turn|junction|road)", out, re.I): return "ordinal"
    heads = set(dirs)
    for d in dirs: heads.update(c for c in ("north", "south", "east", "west") if c in d)   # north-west allows north, west
    plain = out
    for n in sorted(names, key=len, reverse=True):          # "Ring Road South" is not a compass direction
        plain = re.sub(re.escape(n).replace("\\ ", r"[ -]"), " ", plain, flags=re.I)
    for m in DIR_RE.finditer(plain):
        if ndir(m.group(1)) not in heads: return "direction"
    low = out.lower()
    for m in SIDE_RE.finditer(out):
        if not any(m.group(1).lower() in (t or "") for t in turns): return "side"
    if sc.get("avoid") and norm(sc["avoid"]["name"]) and sc["task"].startswith("avoid"):
        if re.search(r"(via|through|past|along|onto|on)\s+(the\s+)?" + re.escape(sc["avoid"]["name"].lower()), low):
            return "uses_avoided"
    return None
