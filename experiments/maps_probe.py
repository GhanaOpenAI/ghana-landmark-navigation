import json, os, sys, requests
KEY = dict(l.strip().split("=", 1) for l in open(".env") if "=" in l)["GEMINI_API_KEY"]
MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemini-2.5-flash"
S = [json.loads(l) for l in open("kumasi_scenarios.jsonl")]
P = {}
for l in open("kumasi_pairs.jsonl"):
    r = json.loads(l); P.setdefault(r["id"], r)
pick = [s for s in S if s["task"] == "route" and s["id"] in P and 2 <= s["route_km"] <= 4][:3]
SYS = ("You are a navigation assistant for Ghana. Give short, spoken, landmark-based directions in simple plain English the way a local "
       "would: name the turns (first/second left), a landmark at each turn so the user knows it is the right one, no distances in numbers, no coordinates.")
for s in pick:
    q = P[s["id"]]["input"]
    body = {"systemInstruction": {"parts": [{"text": SYS}]}, "contents": [{"parts": [{"text": q}]}],
            "tools": [{"googleMaps": {}}],
            "toolConfig": {"retrievalConfig": {"latLng": {"latitude": s["start"]["lat"], "longitude": s["start"]["lon"]}}}}
    r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={KEY}", json=body, timeout=120)
    print("\n" + "=" * 80, "\nQ:", q, f"\n(truth: {s['start']['name']} -> {s['end']['name']}, {s['route_km']} km)")
    if not r.ok: print("HTTP", r.status_code, r.text[:400]); continue
    c = r.json()["candidates"][0]; print("A:", "".join(p.get("text", "") for p in c["content"]["parts"]))
    gm = c.get("groundingMetadata", {}); print("grounding used:", bool(gm), "| chunks:", [x.get("maps", {}).get("title") for x in gm.get("groundingChunks", [])][:6])
    print("TRUE STEPS:", [(st["road"], st.get("turn"), st.get("turn_ordinal"), [x["name"] for x in st.get("turn_lms", [])]) for st in s["steps"]])
