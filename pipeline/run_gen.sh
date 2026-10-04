#!/bin/bash
cd "$(dirname "$0")"
for c in kumasi accra; do WORKERS=32 python3 -u generate_queries.py $c >> gen_$c.log 2>&1; done
echo done > run_gen.done
