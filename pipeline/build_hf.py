"""Assemble the Hugging Face dataset folder (parquet + dataset card) from the pipeline outputs."""
import json, os, csv, shutil
from collections import Counter
import pyarrow as pa, pyarrow.parquet as pq
OUT = "hf_dataset"; shutil.rmtree(OUT, ignore_errors=True)
for d in ("directions", "scenarios", "landmarks"): os.makedirs(f"{OUT}/{d}")
CITIES = ("kumasi", "accra"); stats = {}

# 1) directions: the natural-language pairs (as generated, no capping or de-duplication)
by_split = {"train": [], "test_area": [], "test_pair": []}
for c in CITIES:
    for l in open(f"{c}_pairs.jsonl"):
        r = json.loads(l)
        by_split[r["split"]].append(dict(id=r["id"], city=r["city"], task=r["task"], input=r["input"], output=r["output"]))
for sp, rows in by_split.items():
    pq.write_table(pa.Table.from_pylist(rows), f"{OUT}/directions/{sp}.parquet", compression="zstd")
    stats[sp] = len(rows)
tasks = Counter(r["task"] for rows in by_split.values() for r in rows)
cities = Counter(r["city"] for rows in by_split.values() for r in rows)

# 2) scenarios: the structured facts each pair was written from (nested part kept as JSON text)
n_sc = 0
rows = []
for c in CITIES:
    for l in open(f"{c}_scenarios.jsonl"):
        s = json.loads(l)
        rows.append(dict(id=s["id"], city=s["city"], task=s["task"], split=s["split"], scenario=json.dumps(s, ensure_ascii=False)))
n_sc = len(rows)
pq.write_table(pa.Table.from_pylist(rows), f"{OUT}/scenarios/scenarios.parquet", compression="zstd")

# 3) landmarks: the landmark tables used (OSM + filtered Foursquare), tagged by source
lm = []
for c in CITIES:
    for r in csv.DictReader(open(f"{c}_landmarks.csv")):
        lm.append(dict(city=c.title(), id=r["id"], name=r["name"], lat=float(r["lat"]), lon=float(r["lon"]), category=r["cat"],
                       importance=int(r["imp"]), area=r["area"], source=r.get("source", "osm")))
pq.write_table(pa.Table.from_pylist(lm), f"{OUT}/landmarks/landmarks.parquet", compression="zstd")
src = Counter((x["city"], x["source"]) for x in lm)

card = f"""---
license: odbl
language:
- en
pretty_name: Ghana Landmark Navigation
size_categories:
- 100K<n<1M
task_categories:
- text-generation
tags:
- navigation
- directions
- ghana
- accra
- kumasi
- openstreetmap
- synthetic
configs:
- config_name: directions
  default: true
  data_files:
  - split: train
    path: directions/train.parquet
  - split: test_area
    path: directions/test_area.parquet
  - split: test_pair
    path: directions/test_pair.parquet
- config_name: scenarios
  data_files:
  - split: all
    path: scenarios/scenarios.parquet
- config_name: landmarks
  data_files:
  - split: all
    path: landmarks/landmarks.parquet
---

# Ghana Landmark Navigation

**Author:** [Ghana Open AI](https://huggingface.co/ghanaopenai)

**Supported by** [Ghana NLP](https://ghananlp.org)

Landmark-based, spoken-style directions for **Accra** and **Kumasi**, in simple plain English. People in Ghana rarely give directions as coordinates and kilometres. They say *"after the market, take the second left, you will see the petrol station on your right, that is how you know it is the right turn."* This dataset teaches that style, with the routes grounded in a real road graph.

- **Code that builds it:** https://github.com/GhanaOpenAI/ghana-landmark-navigation
- **{sum(stats.values()):,} request/answer pairs**, built from {n_sc:,} structured route scenarios ({cities['Kumasi']:,} Kumasi pairs, {cities['Accra']:,} Accra pairs).

## Configs

### `directions` (default): the training data
| field | meaning |
|---|---|
| `id` | scenario id; join with the `scenarios` config |
| `city` | Accra or Kumasi |
| `task` | task type (below) |
| `input` | what a user might type |
| `output` | spoken-style directions |

Splits: `train` ({stats['train']:,}), `test_area` ({stats['test_area']:,}), `test_pair` ({stats['test_pair']:,}).
- `test_area`: pairs that touch one of the neighbourhoods held out entirely (about 10% of areas). This is the honest test of generalising to places never seen.
- `test_pair`: about 5% of start/end landmark pairs held out. **Street segments still overlap with training**, so this is an easier test than `test_area`. A pair and its reverse are always in the same split.

Task types: {", ".join(f"`{t}` ({n:,})" for t, n in tasks.most_common())}.
`route`, `reverse` and `short_hop` give directions; `avoid_landmark` and `avoid_road` re-route around something the user wants to skip; `gps_start` is a user who is near a landmark but not at a named place; `nearby` lists landmarks around a place; `connects` answers how two places are linked or whether a landmark is on the way.

### `scenarios`: the structured facts
One row per scenario, with the full structured record as JSON in `scenario`: start, destination, the route as steps (road, which numbered turn it is, side roads skipped, landmarks visible at each turn with their side) and task-specific fields. The answers were written from these facts; use them to audit answers or to build a router-plus-writer system.

### `landmarks`: the landmark tables
{src[('Kumasi','osm')]:,} OSM and {src[('Kumasi','foursquare')]:,} Foursquare landmarks for Kumasi; {src[('Accra','osm')]:,} OSM and {src[('Accra','foursquare')]:,} Foursquare for Accra, with `source`, `category`, `area` and coordinates.

## How it was built
1. Road graph and landmarks from an OpenStreetMap extract (3 Oct 2026), plus filtered Foursquare OS Places to add landmarks (shops, banks, eateries and so on).
2. Scenarios are sampled from the graph by task type. Everything about the route (roads, turn order, counted turns, side roads, landmarks at turns) comes from the graph.
3. Google Gemini (`gemini-3.5-flash-lite`) writes 5 different user requests and answers per scenario from those facts, in plain English. It may add a well-known landmark where the map has none.
4. A fact-checker drops answers that contradict the facts: invented road names, numbered turns the route does not have, wrong left/right, distances or coordinates in the text, over-long or duplicate answers. About {{rej}} of generated pairs were rejected.

**Style rules:** simple English (no pidgin), no kilometres or metres, and **no GPS coordinates anywhere in the text**. People are located by place name, neighbourhood or nearby landmark. The `landmarks` and `scenarios` configs do contain coordinates as map data.

## Limitations
- OpenStreetMap and Foursquare are incomplete and in places out of date. Some landmarks may be closed, mislocated or oddly named (Foursquare entries refreshed as long ago as 2012 to 2019 are included for institutions). Landmarks added by Gemini are not verified.
- Only about 30% of turns have a countable ordinal ("second left"); the others are anchored on landmarks or "keep going". The router ignores OSM turn restrictions.
- The text is synthetic. The facts are checked, but wording and landmark claims at unmapped turns are not guaranteed to be correct. Do not use it for real-world navigation without verification.
- Exact and near duplicates are **not** removed: many scenarios share streets and landmarks.
- English only, two cities.

## Licence and attribution
Map-derived content is built from **© OpenStreetMap contributors** ([ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/)); this dataset is released under ODbL and derived databases must keep attribution and share-alike. Extra landmarks come from [Foursquare OS Places](https://huggingface.co/datasets/foursquare/fsq-os-places) (Apache-2.0). Text was generated with Google Gemini; check Google's terms for generated-output use. Built by [Ghana Open AI](https://huggingface.co/ghanaopenai), supported by [Ghana NLP](https://ghananlp.org).
"""
rej = 0
for c in CITIES: rej += sum(1 for _ in open(f"{c}_rejects.jsonl"))
card = card.replace("{rej}", f"{100*rej/(rej+sum(stats.values())):.0f}%")
open(f"{OUT}/README.md", "w").write(card)
print(stats, "| scenarios", n_sc, "| landmarks", len(lm), "| rejected", rej)
