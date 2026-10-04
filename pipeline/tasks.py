"""Registry of scenario task types. Each sampler: fn(world, rnd) -> list[scenario dict] ([] = failed, retry).

Add a new task by writing a function with @task(name, weight). Scenarios hold ONLY graph-derived facts;
natural language is produced later (generate_queries.py) and checked against these facts (factcheck.py).
"""
import math
import numpy as np
from world import hav, bearing, compass, rnd_m

TASKS = {}


def task(name, weight):
    def deco(fn):
        TASKS[name] = (fn, weight); return fn
    return deco


def pt(l):
    return dict(name=l["name"], lat=l["lat"], lon=l["lon"], area=l["area"], type=l["cat"])


def pair(pool, rnd, dmin, dmax, tries=80):
    for _ in range(tries):
        a, b = rnd.sample(pool, 2)
        if a["node"] == b["node"]: continue
        d = hav(a["lat"], a["lon"], b["lat"], b["lon"])
        if dmin <= d <= dmax: return a, b, d


def base(w, name, a, b, path, steps, d, **extra):
    return dict(task=name, city=w.city.title(), start=pt(a) if "cat" in a else a, end=pt(b),
                straight_km=round(d / 1000, 1), route_km=round(w.plen(path) / 1000, 1),
                bearing=compass(bearing(a["lat"], a["lon"], b["lat"], b["lon"])),
                steps=steps, dest_side=w.dest_side(path, b),
                dest_near=w.nearby(b["lat"], b["lon"], r=100, k=1, exclude={a.get("name"), b["name"]}), **extra)


def detour_ok(w, path, d, k=2.2, slack=800):
    return path and w.plen(path) <= k * d + slack


def sample_route(w, rnd, dmin, dmax, pool=None, min_steps=2):
    p = pair(pool or w.major, rnd, dmin, dmax)
    if not p: return None
    a, b, d = p
    path = w.route(a["node"], b["node"])
    if not detour_ok(w, path, d): return None
    st = w.steps(path, exclude={a["name"], b["name"]})
    if len(st) < min_steps: return None
    return a, b, d, path, st


@task("route", 20)
def route(w, rnd):
    r = sample_route(w, rnd, 1000, 12000)
    if not r: return []
    a, b, d, path, st = r
    return [dict(base(w, "route", a, b, path, st, d), pair=[a["id"], b["id"]])]


@task("reverse", 12)
def reverse(w, rnd):
    """emit A->B and B->A for the same pair so the model sees both directions (one-ways can make them differ)"""
    r = sample_route(w, rnd, 1000, 12000)
    if not r: return []
    a, b, d, path, st = r
    back = w.route(b["node"], a["node"])
    if not detour_ok(w, back, d): return []
    st2 = w.steps(back, exclude={a["name"], b["name"]})
    return [dict(base(w, "route", a, b, path, st, d), pair=[a["id"], b["id"]]),
            dict(base(w, "reverse", b, a, back, st2, d), pair=[a["id"], b["id"]])]


@task("short_hop", 12)
def short_hop(w, rnd):
    p = pair(w.poi, rnd, 150, 800)
    if not p: return []
    a, b, d = p
    path = w.route(a["node"], b["node"])
    if not path or len(path) < 2 or w.plen(path) > 3 * d + 300: return []
    st = w.steps(path, exclude={a["name"], b["name"]})
    return [dict(base(w, "short_hop", a, b, path, st, d), pair=[a["id"], b["id"]])]


@task("avoid_landmark", 12)
def avoid_landmark(w, rnd):
    r = sample_route(w, rnd, 2000, 12000, min_steps=3)
    if not r: return []
    a, b, d, path, st = r
    mids = [n for s in st[:-1] for n in s["landmarks_at_end"] if n not in (a["name"], b["name"])]
    if not mids: return []
    name = rnd.choice(mids)
    lm = next((l for l in w.poi if l["name"] == name), None)
    if not lm: return []
    banned = w.nodes_within(lm["lat"], lm["lon"], 150)
    if a["node"] in banned or b["node"] in banned: return []
    alt = w.route(a["node"], b["node"], banned_nodes=banned)
    if not alt or alt == path or w.plen(alt) > 2.5 * w.plen(path) or w.plen(alt) < 1.02 * w.plen(path): return []
    st2 = w.steps(alt, exclude={a["name"], b["name"], name})
    return [dict(base(w, "avoid_landmark", a, b, alt, st2, d), pair=[a["id"], b["id"]],
                 avoid=dict(kind="landmark", name=name), original_km=round(w.plen(path) / 1000, 1))]


@task("avoid_road", 8)
def avoid_road(w, rnd):
    r = sample_route(w, rnd, 2000, 12000, min_steps=3)
    if not r: return []
    a, b, d, path, st = r
    roads = [s["road"] for s in st if not s["generic"] and s["dist_m"] >= 300]
    if not roads: return []
    road = rnd.choice(roads)
    alt = w.route(a["node"], b["node"], banned_road=road)
    if not alt or alt == path or w.plen(alt) > 2.5 * w.plen(path): return []
    st2 = w.steps(alt, exclude={a["name"], b["name"]})
    if any(s["road"] == road for s in st2): return []
    return [dict(base(w, "avoid_road", a, b, alt, st2, d), pair=[a["id"], b["id"]],
                 avoid=dict(kind="road", name=road), original_km=round(w.plen(path) / 1000, 1))]


def random_point(w, rnd):
    la, lo = w.coord[rnd.choice(w.nodes)]
    return la + rnd.uniform(-3e-4, 3e-4), lo + rnd.uniform(-3e-4, 3e-4)


def gps_point(w, rnd, la, lo, near):
    return dict(name=None, lat=round(la, 5), lon=round(lo, 5), area=w.area_of(la, lo), type="gps point", near=near)


@task("gps_start", 14)
def gps_start(w, rnd):
    la, lo = random_point(w, rnd)
    near = w.nearby(la, lo, r=400, k=1)
    b = rnd.choice(w.major)
    d = hav(la, lo, b["lat"], b["lon"])
    if not near or not 800 <= d <= 9000: return []
    src, off = w.snap(la, lo)
    path = w.route(src, b["node"])
    if off > 80 or not detour_ok(w, path, d): return []
    st = w.steps(path, exclude={b["name"]})
    s = gps_point(w, rnd, la, lo, near[0])
    return [dict(base(w, "gps_start", s, b, path, st, d), pair=[f"gps:{s['area']}", b["id"]])]


@task("nearby", 8)
def nearby(w, rnd):
    if rnd.random() < 0.5:
        l = rnd.choice(w.major); s = pt(l); la, lo = l["lat"], l["lon"]; excl = {l["name"]}
    else:
        la, lo = random_point(w, rnd); s = gps_point(w, rnd, la, lo, None); excl = set()
    near = w.nearby(la, lo, r=800, k=6, exclude=excl)
    if len(near) < 3: return []
    near.sort(key=lambda x: x["dist_m"])
    return [dict(task="nearby", city=w.city.title(), start=s, nearby=near, end=None, pair=[s["name"] or f"gps:{s['area']}"])]


@task("connects", 8)
def connects(w, rnd):
    r = sample_route(w, rnd, 1500, 10000, min_steps=3)
    if not r: return []
    a, b, d, path, st = r
    sc = dict(base(w, "connects", a, b, path, st, d), pair=[a["id"], b["id"]])
    on = [(s, n) for s in st for n in s["landmarks_at_end"]]
    if rnd.random() < 0.5 and on:
        s, n = rnd.choice(on)
        sc.update(variant="on_the_way", question_landmark=n, on_route=True, on_road=s["road"])
    else:
        # landmark that is NOT on the route: 500-2000 m from every route node
        xy = np.array([w.coord[n] for n in path])
        cands = [l for l in rnd.sample(w.major, min(60, len(w.major))) if l["name"] not in (a["name"], b["name"])]
        for l in cands:
            dm = float(np.min(w._d(xy, l["lat"], l["lon"])))
            if 500 <= dm <= 2000 and all(l["name"] not in s["landmarks_at_end"] for s in st):
                sc.update(variant="on_the_way", question_landmark=l["name"], on_route=False, off_route_m=rnd_m(dm)); break
        else:
            sc.update(variant="link")
    sc.setdefault("variant", "link")
    return [sc]
