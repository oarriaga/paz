#!/bin/bash

SEED=10

for i in $(seq 1 20)
do
  echo "--- Starting Experiment Run #${i} ---"
  for WAYS in $(seq 100 50 900)
  do
    LABEL="${WAYS}-WAY_RUN-${i}_SEED-${SEED}"
    echo "Running training with: --label \"$LABEL\" --seed $SEED --train_classes $WAYS"
    python3 train.py \
      --label "$LABEL" \
      --seed "$SEED" \
      --train_classes "$WAYS" \
      --validation_split 0.0
    SEED=$((SEED + 1))
  done
  echo "--- Finished Experiment Run #${i} ---"
done

echo "All training runs completed."
