#!/bin/bash
# usage: LIMIT=3000 EVAL_N=200 TAG=bo ./bakeoff.sh   -- trains+evals each model sequentially with identical settings
cd "$(dirname "$0")"; source .venv/bin/activate
MODELS=${MODELS:-"Qwen/Qwen3.5-0.8B openbmb/MiniCPM5-1B google/gemma-3-1b-it Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B google/gemma-4-E4B-it"}
for m in $MODELS; do
  n=$(echo $m | tr '/' '_')
  echo "=== $m $(date +%T)" >> ${TAG:-bo}.log
  python train.py --model $m --out runs/${TAG:-bo}_$n --limit ${LIMIT:-3000} --epochs ${EPOCHS:-1} >> logs_${TAG:-bo}_$n.train 2>&1 \
    && python eval.py --model $m --adapter runs/${TAG:-bo}_$n --n ${EVAL_N:-200} --out results/${TAG:-bo}_$n.json >> logs_${TAG:-bo}_$n.eval 2>&1 \
    && echo "ok $m" >> ${TAG:-bo}.log || echo "FAILED $m (see logs_${TAG:-bo}_$n.*)" >> ${TAG:-bo}.log
done
echo finished >> ${TAG:-bo}.log
