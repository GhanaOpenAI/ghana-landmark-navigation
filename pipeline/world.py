"""Routable world for one city: drivable road graph + landmarks, parsed from an OSM extract (cached)."""
import math, os, pickle
from collections import defaultdict
import numpy as np
import networkx as nx
import osmium
from rapidfuzz import fuzz

DRIVE = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
         "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link", "living_street"}
# unnamed roads are described by class instead of "unnamed road"
LABEL = {"motorway": "the highway", "trunk": "the highway", "primary": "the main road",
         "secondary": "the main road", "tertiary": "a connecting road", "unclassified": "a local road",
         "residential": "a side street", "living_street": "a side street"}
MAX_STEPS = 5      # longer routes get their smallest steps merged away
MIN_STEP_M = 120   # steps shorter than this are merged into a neighbour
COMP = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]


def categorize(t):
    """OSM tags -> (category, importance 2-5) or None. Importance ~ how likely people name it in directions."""
    a = t.get("amenity")
    if a == "marketplace": return "market", 5
    if a == "bus_station": return "lorry park", 5
    if a == "hospital": return "hospital", 4
    if a == "police": return "police station", 3
    if a == "place_of_worship": return ("mosque", 2) if t.get("religion") == "muslim" else ("church", 2)
    if a in ("school", "university", "college"): return "school", 3
    if a == "bank": return "bank", 2
    if a == "fuel": return "filling station", 2
    if t.get("junction") == "roundabout": return "roundabout", 4
    if t.get("tourism") in ("attraction", "museum", "hotel"): return t["tourism"], 3
    if t.get("leisure") in ("stadium", "park"): return t["leisure"], 4
    if t.get("shop") == "mall": return "mall", 4
    if t.get("railway") == "station": return "railway station", 4
    if a == "pharmacy": return "pharmacy", 2
    if a in ("restaurant", "fast_food", "cafe", "bar", "pub"): return "eatery", 2
    if a in ("clinic", "doctors", "dentist"): return "clinic", 2
    if a in ("post_office", "townhall", "community_centre", "courthouse", "fire_station"): return a.replace("_", " "), 3
    if a in ("atm",): return "ATM", 2
    if t.get("highway") == "bus_stop" or t.get("public_transport") == "station": return "bus stop", 2
    if t.get("shop"): return "shop", 2
    if t.get("building") and t.get("name"): return "building", 2
    if t.get("place") in ("suburb", "neighbourhood", "town", "quarter"): return "area", 4
    return None


def hav(a, b, c, d):
    p = math.pi / 180
    x = math.sin((c - a) * p / 2) ** 2 + math.cos(a * p) * math.cos(c * p) * math.sin((d - b) * p / 2) ** 2
    return 12742000 * math.asin(math.sqrt(x))


def bearing(a, b, c, d):
    p = math.pi / 180
    y = math.sin((d - b) * p) * math.cos(c * p)
    x = math.cos(a * p) * math.sin(c * p) - math.sin(a * p) * math.cos(c * p) * math.cos((d - b) * p)
    return (math.atan2(y, x) / p + 360) % 360


def compass(b): return COMP[int((b + 22.5) // 45) % 8]


def rnd_m(m):
    return int(round(m, -1) if m < 1000 else round(m / 50) * 50) or 10


class _Parse(osmium.SimpleHandler):
    def __init__(s):
        super().__init__(); s.ways = []; s.lm = []

    def node(s, n):
        c = categorize(n.tags)
        if c and n.tags.get("name"):
            s.lm.append((n.tags["name"], n.location.lat, n.location.lon, c[0], c[1], n.tags.get("addr:suburb", "")))

    def way(s, w):
        t = w.tags; hw = t.get("highway")
        if hw in DRIVE:
            try: pts = [(x.ref, x.lat, x.lon) for x in w.nodes]
            except osmium.InvalidLocationError: return
            s.ways.append((pts, t.get("name") or t.get("ref") or "", hw, t.get("oneway") == "yes"))
            if t.get("junction") == "roundabout" and t.get("name"):
                s.lm.append((t["name"], pts[0][1], pts[0][2], "roundabout", 4, ""))
        else:
            c = categorize(t)
            if c and t.get("name"):
                try: la = [x.lat for x in w.nodes]; lo = [x.lon for x in w.nodes]
                except osmium.InvalidLocationError: return
                s.lm.append((t["name"], sum(la) / len(la), sum(lo) / len(lo), c[0], c[1], t.get("addr:suburb", "")))


class World:
    def __init__(self, city, G, coord, LM):
        self.city, self.G, self.coord = city, G, coord
        self.nodes = list(G.nodes)
        self.arr = np.array([coord[n] for n in self.nodes])
        self.LM = LM
        self.areas = [l for l in LM if l["cat"] == "area"]
        self.poi = [l for l in LM if l["cat"] != "area"]
        self.major = [l for l in self.poi if l["imp"] >= 3]
        self.poi_xy = np.array([[l["lat"], l["lon"]] for l in self.poi])
        for l in LM:
            l["area"] = l["name"] if l["cat"] == "area" else (l["suburb"] or self.area_of(l["lat"], l["lon"]))

    # ---- geometry helpers
    @staticmethod
    def _d(arr, la, lo):
        return np.hypot((arr[:, 0] - la) * 111000, (arr[:, 1] - lo) * 111000 * math.cos(la * math.pi / 180))

    def area_of(self, la, lo):
        if not self.areas: return ""
        return min(self.areas, key=lambda a: hav(la, lo, a["lat"], a["lon"]))["name"]

    def snap(self, la, lo):
        d = self._d(self.arr, la, lo); j = int(d.argmin())
        return self.nodes[j], float(d[j])

    def nodes_within(self, la, lo, r):
        return set(self.nodes[i] for i in np.where(self._d(self.arr, la, lo) <= r)[0])

    def nearby(self, la, lo, r=200, k=2, exclude=(), min_imp=2):
        """landmarks within r metres, most important first -> list of dict(name, type, dist_m, direction)"""
        d = self._d(self.poi_xy, la, lo); out = []
        for i in np.argsort(d)[:25]:
            l = self.poi[i]
            if d[i] <= r and l["imp"] >= min_imp and l["name"] not in exclude:
                out.append((l, float(d[i])))
        out.sort(key=lambda x: (-x[0]["imp"], x[1]))
        return [dict(name=l["name"], type=l["cat"], dist_m=rnd_m(di),
                     direction=compass(bearing(la, lo, l["lat"], l["lon"]))) for l, di in out[:k]]

    # ---- routing
    def route(self, src, dst, banned_nodes=(), banned_road=None):
        if not banned_nodes and not banned_road:
            w = "length"
        else:
            bn = set(banned_nodes)
            w = lambda u, v, e: None if (u in bn or v in bn or (banned_road and e["name"] == banned_road)) else e["length"]
        try: return nx.shortest_path(self.G, src, dst, weight=w)
        except (nx.NetworkXNoPath, nx.NodeNotFound): return None

    def plen(self, p): return sum(self.G[u][v]["length"] for u, v in zip(p, p[1:]))

    def steps(self, path, exclude=()):
        """Path -> human-scale steps: unnamed roads labelled by class, tiny steps merged, <= MAX_STEPS."""
        G, co = self.G, self.coord
        segs = []
        for u, v in zip(path, path[1:]):
            e = G[u][v]; named = bool(e["name"]); road = e["name"] or LABEL.get(e["hw"], "a side street")
            b = bearing(*co[u], *co[v])
            if segs and segs[-1]["road"] == road:
                s = segs[-1]; s["dist"] += e["length"]; s["end"] = v; s["last_b"] = b
            else:
                segs.append(dict(road=road, named=named, dist=e["length"], start=u, end=v, first_b=b, last_b=b))

        def merge(i):
            j = i - 1 if i > 0 else i + 1
            lo = min(i, j); a, b = segs[lo], segs[lo + 1]
            tot = a["dist"] + b["dist"]
            named = [x for x in (a, b) if x["named"] and x["dist"] >= 0.3 * tot]   # prefer a real road name
            keep = max(named or (a, b), key=lambda x: x["dist"])
            m = dict(road=keep["road"], named=keep["named"], dist=a["dist"] + b["dist"], start=a["start"],
                     end=b["end"], first_b=a["first_b"], last_b=b["last_b"])
            segs[lo:lo + 2] = [m]
            while lo + 1 < len(segs) and segs[lo + 1]["road"] == m["road"]:   # re-collapse same road
                n = segs.pop(lo + 1); m["dist"] += n["dist"]; m["end"] = n["end"]; m["last_b"] = n["last_b"]

        GENERIC_MIN_M = 300   # generic ("a side street") stretches only survive if they are substantial
        while len(segs) > 1:
            weak = [k for k, g in enumerate(segs) if g["dist"] < MIN_STEP_M or (not g["named"] and g["dist"] < GENERIC_MIN_M)]
            if weak: merge(min(weak, key=lambda k: segs[k]["dist"]))
            elif len(segs) > MAX_STEPS: merge(min(range(len(segs)), key=lambda k: (segs[k]["named"], segs[k]["dist"])))
            else: break
        out, used = [], set(exclude)
        for i, s in enumerate(segs):
            b = bearing(*co[s["start"]], *co[s["end"]])
            st = dict(road=s["road"], generic=not s["named"], heading=compass(b), dist_m=rnd_m(s["dist"]),
                      start_node=s["start"], end_node=s["end"])
            if i:
                d = (s["first_b"] - segs[i - 1]["last_b"] + 540) % 360 - 180
                st["turn"] = ("straight" if abs(d) < 25 else "slight right" if 25 <= d < 60 else "right" if 60 <= d < 135
                              else "sharp right" if d >= 135 else "slight left" if -60 < d <= -25
                              else "left" if -135 < d <= -60 else "sharp left")
            lms = [x for x in self.nearby(*co[s["end"]], r=200, k=3, exclude=used)][:2]
            used.update(x["name"] for x in lms)
            st["landmarks_at_end"] = [x["name"] for x in lms]
            out.append(st)
        self._annotate(path, out, set(exclude))
        return out

    def _side_roads(self, path, i):
        """sides of the small roads branching off at path[i] (excluding the route's own prev/next)"""
        if not 0 < i < len(path) - 1: return []
        G, co = self.G, self.coord
        n, prev, nxt = path[i], path[i - 1], path[i + 1]; b = bearing(*co[prev], *co[n]); out = []
        for m in set(G.successors(n)) | set(G.predecessors(n)):
            if m in (prev, nxt): continue
            d = (bearing(*co[n], *co[m]) - b + 540) % 360 - 180
            if 25 < d < 155: out.append("right")
            elif -155 < d < -25: out.append("left")
        return out

    def _lms_at(self, la, lo, b_in, r=120, k=3, exclude=()):
        """landmarks within r m of a junction, best first, each with the side of the road it is on"""
        d = self._d(self.poi_xy, la, lo); out = []
        for i in np.argsort(d)[:12]:
            l = self.poi[i]
            if d[i] > r or l["name"] in exclude: continue
            rel = (bearing(la, lo, l["lat"], l["lon"]) - b_in + 540) % 360 - 180
            out.append((l, float(d[i]), "right" if 15 < rel < 165 else "left" if -165 < rel < -15 else "ahead"))
        out.sort(key=lambda x: (-x[0]["imp"], x[1]))
        keep = []
        for l, di, sd in out:
            if all(not (abs(l["lat"] - o["lat"]) < 0.0004 and abs(l["lon"] - o["lon"]) < 0.0004) and
                   fuzz.partial_token_set_ratio(l["name"].lower(), o["name"].lower()) < 70 for o, _, _ in keep): keep.append((l, di, sd))
        out = keep
        return [dict(name=l["name"], type=l["cat"], side=sd, dist_m=rnd_m(di), source=l.get("source", "osm")) for l, di, sd in out[:k]]

    def _annotate(self, path, out, exclude):
        """street-level turn facts: which numbered turn it is, side roads skipped, landmarks visible at the turn"""
        idx = {n: i for i, n in enumerate(path)}; co = self.coord
        for k, t in enumerate(out):
            i0 = idx[t["start_node"]]
            if k == 0:
                t["side_roads_passed"] = sum(len(self._side_roads(path, j)) for j in range(1, idx[t["end_node"]]))
                continue
            p0 = idx[out[k - 1]["start_node"]]
            passed = [x for j in range(p0 + 1, i0 + 1) for x in self._side_roads(path, j)]
            turn = t["turn"]; side = "left" if "left" in turn else "right" if "right" in turn else None
            t["side_roads_passed"] = len(passed)
            if side:
                n = passed.count(side) + 1
                t["turn_ordinal"] = n if n <= 3 else None     # "sixth left" is useless; None => anchor on a landmark
            la, lo = co[t["start_node"]]
            t["turn_lms"] = self._lms_at(la, lo, bearing(*co[path[max(i0 - 1, 0)]], la, lo), exclude=exclude)[:2]

    def dest_side(self, path, dest):
        """which side of the final stretch the destination is on"""
        if len(path) < 2: return "ahead"
        a, b = self.coord[path[-2]], self.coord[path[-1]]
        if hav(*b, dest["lat"], dest["lon"]) < 25: return "ahead"
        d = (bearing(*b, dest["lat"], dest["lon"]) - bearing(*a, *b) + 540) % 360 - 180
        return "right" if 20 < d < 160 else "left" if -160 < d < -20 else "ahead"


def load_world(city):
    pbf, pkl = f"{city}.osm.pbf", f"{city}_world.pkl"
    if os.path.exists(pkl) and os.path.getmtime(pkl) > os.path.getmtime(pbf):
        return pickle.load(open(pkl, "rb"))
    h = _Parse(); h.apply_file(pbf, locations=True)
    cnt = defaultdict(int); coord = {}
    for pts, *_ in h.ways:
        for i, (r, la, lo) in enumerate(pts):
            coord[r] = (la, lo); cnt[r] += 2 if i in (0, len(pts) - 1) else 1
    G = nx.DiGraph()
    for pts, name, hw, ow in h.ways:
        last = 0
        for i in range(1, len(pts)):
            if cnt[pts[i][0]] >= 2 or i == len(pts) - 1:
                seg = pts[last:i + 1]
                L = sum(hav(seg[k][1], seg[k][2], seg[k + 1][1], seg[k + 1][2]) for k in range(len(seg) - 1))
                a, b = seg[0][0], seg[-1][0]
                if a != b:
                    G.add_edge(a, b, name=name, hw=hw, length=L)
                    if not ow: G.add_edge(b, a, name=name, hw=hw, length=L)
                last = i
    G = G.subgraph(max(nx.strongly_connected_components(G), key=len)).copy()
    nodes = list(G.nodes); arr = np.array([coord[n] for n in nodes])
    seen = {}
    for nm, la, lo, cat, imp, sub in h.lm:
        k = nm.strip().lower()
        if len(k) < 3 or (k in seen and seen[k][4] >= imp): continue
        seen[k] = (nm.strip(), la, lo, cat, imp, sub)
    LM = []
    for i, (nm, la, lo, cat, imp, sub) in enumerate(seen.values()):
        d = World._d(arr, la, lo); j = int(d.argmin())
        if d[j] > 250: continue            # not reachable by road
        LM.append(dict(id=f"{city}_{i}", name=nm, lat=round(la, 5), lon=round(lo, 5), cat=cat, imp=imp,
                       node=nodes[j], suburb=sub, source="osm"))
    try:                                        # enrich with filtered Foursquare places (Apache-2.0)
        import fsq
        bbox = (float(arr[:, 0].min()), float(arr[:, 0].max()), float(arr[:, 1].min()), float(arr[:, 1].max()))
        osm_poi = [l for l in LM if l["cat"] != "area"]; taken = {l["name"].lower() for l in LM}
        for nm, la, lo, cat, imp in fsq.load(city, bbox, osm_poi):
            if nm.lower() in taken: continue
            d = World._d(arr, la, lo); j = int(d.argmin())
            if d[j] > 250: continue
            taken.add(nm.lower())
            LM.append(dict(id=f"{city}_f{len(LM)}", name=nm, lat=round(la, 5), lon=round(lo, 5), cat=cat, imp=imp,
                           node=nodes[j], suburb="", source="foursquare"))
    except ImportError:
        pass
    w = World(city, G, {n: coord[n] for n in G.nodes}, LM)
    pickle.dump(w, open(pkl, "wb"))
    return w
