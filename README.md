# Ghana Landmark Navigation

**Author:** [Ghana Open AI](https://huggingface.co/ghanaopenai)

**Supported by** [Ghana NLP](https://ghananlp.org)

**Dataset:** [`ghanaopenai/ghana-landmark-navigation`](https://huggingface.co/datasets/ghanaopenai/ghana-landmark-navigation) on Hugging Face (291,412 request/answer pairs for Accra and Kumasi).

Pipeline for building a training dataset of **landmark-based, spoken-style directions for Ghana** (currently Accra and Kumasi), and for training and evaluating small language models on it.

People in Ghana rarely give directions as coordinates and kilometres. They say *"after the market, take the second left — you'll see the petrol station on your right, that's how you know it's the right turn."* This project turns real map data into that style of text, grounded in facts from a road graph.

## How it works

```
OpenStreetMap extract + filtered Foursquare places
        │
   world.py          road graph + landmark table (source-tagged: osm / foursquare)
        │
   tasks.py          8 task types sampled from the graph  ──►  structured scenarios (facts only)
   build_scenarios.py   (held-out split by area and by pair)
        │
   generate_queries.py   Gemini writes 5 natural user requests + answers per scenario,
        │               given the facts as a "skeleton" (road, counted turn, landmarks seen at the turn)
   factcheck.py      rejects answers that contradict the skeleton
        │
   train/            SFT + evaluation of small models (Qwen3.5, Gemma, MiniCPM5)
```

**Scenario facts come from the graph, not from the LLM.** Gemini only turns the skeleton into natural wording. Each step carries the road, which numbered turn it is ("second left", only up to three), how many side roads are skipped, and the landmarks visible at the turn with their side of the road.

### Task types
`route`, `reverse`, `short_hop`, `avoid_landmark`, `avoid_road`, `gps_start` (user is near a landmark, not at a named place), `nearby`, `connects` (link / "is X on the way?"). Add a new one with `@task(name, weight)` in `pipeline/tasks.py`.

### Style rules for the generated answers
Simple plain English (no pidgin), spoken style, **no distances in numbers and no GPS coordinates anywhere in the text**, inputs included (people are located by place name, neighbourhood or nearby landmark). Where the map has no landmark at a turn, the answer relies on the counted turn and a natural hedge instead of inventing one.

### Fact-check (`factcheck.py`)
Drops generated pairs that: contain coordinates or the word GPS; name a road not in the facts; state a numbered turn the route doesn't have; use a left/right or compass word the route doesn't contain; include distances or coordinates in the answer; are over-long or duplicates; or route through a place the user asked to avoid. Landmarks are **not** validated against OSM (the map is incomplete and Gemini is allowed to add well-known ones).

## Running it

```bash
pip install -r requirements.txt
cd pipeline
# 1. a Geofabrik Ghana extract -> city boxes
osmium extract -b -1.75,6.58,-1.45,6.82 ghana-latest.osm.pbf -o kumasi.osm.pbf
osmium extract -b -0.36,5.48,-0.08,5.74 ghana-latest.osm.pbf -o accra.osm.pbf
# 2. (optional) Foursquare OS Places Ghana rows -> ghana_fsq.parquet  (needs HF access to the gated dataset)
python fsq_ghana.py
# 3. scenarios
python build_scenarios.py kumasi 32000
# 4. natural language (needs GEMINI_API_KEY in the environment or a .env file; never commit it)
WORKERS=32 python generate_queries.py kumasi      # resumable; clean_leaks.py re-queues scenarios with bad inputs
# 5. training data + bake-off (GPU)
cd ../train && python prep_data.py kumasi accra && LIMIT=3000 EVAL_N=200 TAG=bo ./bakeoff.sh
```

## Findings so far
- First bake-off (Kumasi, 3,000 examples, 1 epoch, memorise-the-streets setup — answers written from the user request alone): all of Qwen3.5-0.8B/2B/4B, Gemma-3-1B, Gemma-4-E4B and MiniCPM5-1B write fluent, correctly styled directions but **invent turn counts and road names for routes they have not seen** (pass rates 5–20%). Model size did not fix this at this training size. That run used an earlier, pidgin-heavy version of the data.
- Direction of travel: use a **router + verbaliser** design. The graph produces the route skeleton at inference time and a small model (0.5–1B) writes the directions, so the model does not have to memorise street layouts.
- Gemini with the Google Maps grounding tool is a place lookup, not a router: it refused or invented turn-by-turn directions, so it is not used as a route source.

## Data sources and licences
- Road graph and landmarks: © OpenStreetMap contributors, **ODbL** — derived data needs attribution and share-alike.
- Extra landmarks: [Foursquare OS Places](https://huggingface.co/datasets/foursquare/fsq-os-places), **Apache-2.0**, filtered (open places only, plausible categories, de-duplicated against OSM). Entries are tagged `source=foursquare`; some are old (refreshed 2012–2019).
- Natural-language text: generated with Google Gemini from the structured scenarios.

## Caveats
- OSM and Foursquare data are incomplete and in places out of date; a few landmarks will be closed or wrong. The router ignores OSM turn restrictions.
- Only ~30% of turns have a countable ordinal; the rest are anchored on landmarks or "keep going".
- Held-out `test_pair` splits hold out start/end pairs, not street segments, so segments overlap with training; `test_area` (whole neighbourhoods held out) is the honest generalisation test.

## Status
The dataset is published: 261,007 train, 16,716 `test_area` and 13,689 `test_pair` pairs (Accra and Kumasi, 32,000 scenarios each). `pipeline/build_hf.py` assembles the Hugging Face release. Training work (router + verbaliser) is next.

---
Built by [Ghana Open AI](https://huggingface.co/ghanaopenai), supported by [Ghana NLP](https://ghananlp.org).
