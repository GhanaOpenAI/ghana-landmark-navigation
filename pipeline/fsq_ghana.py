"""Pull Ghana rows from Foursquare OS Places (Apache-2.0).  Runs where the network is fast (H200)."""
import os, sys, pyarrow.dataset as ds, pyarrow.compute as pc, pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
fs = HfFileSystem(token=open(os.path.expanduser("~/.cache/huggingface/token")).read().strip())
base = "datasets/foursquare/fsq-os-places/release/dt=2026-09-15/places/parquet"
files = sorted(f for f in fs.ls(base, detail=False) if f.endswith(".parquet"))
print(len(files), "place files"); print(pq.ParquetFile(fs.open(files[0])).schema_arrow.names)
d = ds.dataset(files, filesystem=fs, format="parquet")
cols = [c for c in ["fsq_place_id", "name", "latitude", "longitude", "address", "locality", "region", "country",
                    "fsq_category_labels", "date_created", "date_refreshed", "date_closed"] if c in d.schema.names]
t = d.to_table(columns=cols, filter=pc.field("country") == "GH")
pq.write_table(t, "ghana_fsq.parquet"); print("Ghana rows:", t.num_rows, "| columns:", cols)
