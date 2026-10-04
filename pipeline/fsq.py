"""Filtered Foursquare OS Places (Apache-2.0) as extra landmarks. Keeps only things people name in directions."""
import re, os
import numpy as np
from rapidfuzz import fuzz

# (keyword in Foursquare category label, our category, importance)
FSQ_MAP = [("Bus Station", "lorry park", 5), ("Market", "market", 5), ("Shopping Mall", "mall", 4), ("Hospital", "hospital", 4),
           ("Stadium", "stadium", 4), ("Supermarket", "supermarket", 3), ("Hotel", "hotel", 3), ("Lodging", "hotel", 3),
           ("School", "school", 3), ("College", "school", 3), ("University", "school", 3), ("Police Station", "police station", 3),
           ("Post Office", "post office", 3), ("Monument", "monument", 3), ("Park", "park", 3), ("Bank", "bank", 2),
           ("ATM", "ATM", 2), ("Gas Station", "filling station", 2), ("Fuel", "filling station", 2), ("Pharmacy", "pharmacy", 2),
           ("Clinic", "clinic", 2), ("Church", "church", 2), ("Mosque", "mosque", 2), ("Restaurant", "eatery", 2),
           ("Café", "eatery", 2), ("Coffee", "eatery", 2), ("Bar", "eatery", 2), ("Fast Food", "eatery", 2), ("Bus Stop", "bus stop", 2)]
INSTITUTIONS = {"lorry park", "market", "mall", "hospital", "stadium", "supermarket", "hotel", "school", "police station",
                "post office", "monument", "park", "bank", "filling station", "church", "mosque"}
BAD = re.compile(r"\b(room|gate|block|dept|department|unit|office|lab|laboratory|hostel|flat|floor|annex|dormitory|lecture|hall \d)\b|,|\d{3,}", re.I)
CITIES = {"accra", "kumasi", "takoradi", "tamale", "tema", "cape coast", "sunyani", "koforidua", "ho"}


def classify(labels):
    for lab in (labels if labels is not None else []):
        for kw, cat, imp in FSQ_MAP:
            if kw.lower() in str(lab).lower(): return cat, imp
    return None


def load(city, bbox, osm_poi, path="ghana_fsq.parquet"):
    """-> list of (name, lat, lon, cat, imp) for places not already an OSM landmark"""
    if not os.path.exists(path): return []
    import pyarrow.parquet as pq
    t = pq.read_table(path).to_pandas()
    la0, la1, lo0, lo1 = bbox
    t = t[t.date_closed.isna() & t.name.notna() & t.latitude.between(la0, la1) & t.longitude.between(lo0, lo1)]
    other = CITIES - {city.lower()}
    xy = np.array([[l["lat"], l["lon"]] for l in osm_poi]); onames = [l["name"].lower() for l in osm_poi]
    out, seen = [], set()
    for r in t.itertuples():
        nm = " ".join(str(r.name).split())
        low = nm.lower()
        c = classify(r.fsq_category_labels)
        if not c or len(nm) < 4 or BAD.search(nm) or any(re.search(rf"\b{o}\b", low) for o in other): continue
        yr = int(str(r.date_refreshed)[:4]) if r.date_refreshed is not None else 0
        if yr < 2016 and c[0] not in INSTITUTIONS: continue          # old shops/eateries are likely gone; institutions persist
        if low in seen: continue
        d = np.hypot((xy[:, 0] - r.latitude) * 111000, (xy[:, 1] - r.longitude) * 111000 * 0.99)
        if any(fuzz.token_set_ratio(low, onames[i]) >= 85 for i in np.where(d < 150)[0]): continue   # already in OSM
        if any(abs(r.latitude - o[1]) < 0.0012 and abs(r.longitude - o[2]) < 0.0012 and
               fuzz.partial_token_set_ratio(re.sub(r"\(.*?\)", "", low), re.sub(r"\(.*?\)", "", o[0].lower())) >= 80 for o in out): continue
        seen.add(low); out.append((nm, r.latitude, r.longitude, c[0], c[1]))
    return out
