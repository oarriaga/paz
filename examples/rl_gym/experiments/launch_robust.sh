#!/bin/bash
# Full replication of the original robust walking run:
# 2 ranks x 4096 envs, 30000 iterations, adaptive LR from 1e-3.
cd "$(dirname "$0")/.."
PY=$HOME/miniconda3/envs/mobi/bin/python
ARGS="--num_envs 4096 --num_iterations 30000 --num_processes 2"
setsid nohup $PY -u learn_to_walk.py $ARGS --process_id 1 \
    > experiments/robust_rank1.log 2>&1 &
echo "rank1 pid $!"
setsid nohup $PY -u learn_to_walk.py $ARGS --process_id 0 \
    > experiments/robust_rank0.log 2>&1 &
echo "rank0 pid $!"
