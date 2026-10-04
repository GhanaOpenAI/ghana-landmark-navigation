import json, sys, requests
KEY = dict(l.strip().split("=", 1) for l in open(".env") if "=" in l)["GEMINI_API_KEY"]
S = [json.loads(l) for l in open("kumasi_scenarios.jsonl")]
pick = [s for s in S if s["task"] == "route" and s["start"]["name"] and 2 <= s["route_km"] <= 4 and s["steps"][1:] and any(st.get("turn_ordinal") for st in s["steps"])][:2]
SYS = ("You help people find their way around Ghana. You have Google Maps place search. Use it to identify the start and destination, "
       "then give directions the way a local would: simple plain English, 'after the market take the second left, you'll see the bank on your right'. "
       "Use whatever you know about the area and the places you found. If you are unsure of a detail, say so honestly and give your best guidance. No numeric distances.")
for MODEL in ("gemini-2.5-flash", "gemini-3.5-flash"):
    for s in pick:
        q = f"How do I get from {s['start']['name']} to {s['end']['name']} in {s['city']}?"
        body = {"systemInstruction": {"parts": [{"text": SYS}]}, "contents": [{"parts": [{"text": q}]}], "tools": [{"googleMaps": {}}],
                "toolConfig": {"retrievalConfig": {"latLng": {"latitude": s["start"]["lat"], "longitude": s["start"]["lon"]}}}}
        r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={KEY}", json=body, timeout=180)
        print("\n" + "=" * 90, f"\n[{MODEL}] Q:", q)
        if not r.ok: print("HTTP", r.status_code, r.text[:300]); continue
        c = r.json()["candidates"][0]; print("A:", "".join(p.get("text", "") for p in c["content"]["parts"])[:900])
        print("TRUE:", [(st["road"], st.get("turn"), st.get("turn_ordinal"), [x["name"] for x in st.get("turn_lms", [])][:2]) for st in s["steps"]])
