#!/bin/bash

# python3 train.py --train_classes 12 --train_ways 5 --train_shots 5 --train_queries 1 --validation_split 0.5 --steps_per_epoch 10 --epochs 2
# python3 train.py --label "6-WAY_SEED-10" --seed 10 --train_classes 6 --train_ways 5 --train_shots 5 --train_queries 1 --validation_split 0.0
# python3 train.py --label "8-WAY_SEED-11" --seed 11 --train_classes 8 --train_ways 5 --train_shots 5 --train_queries 1 --validation_split 0.0
# python3 train.py --label "10-WAY_SEED-12" --seed 12 --train_classes 10 --train_ways 5 --train_shots 5 --train_queries 1 --validation_split 0.0
# python3 train.py --label "12-WAY_SEED-13"  --seed 13 --train_classes 12 --train_ways 5 --train_shots 5 --train_queries 1 --validation_split 0.0
# python3 train.py --label "14-WAY_SEED-14"  --seed 14 --train_classes 14 --train_ways 5 --train_shots 5 --train_queries 1 --validation_split 0.0
# python3 train.py --label "16-WAY_SEED-15"  --seed 15 --train_classes 16 --train_ways 5 --train_shots 5 --train_queries 1 --validation_split 0.0
# python3 train.py --label "18-WAY_SEED-16"  --seed 16 --train_classes 18 --train_ways 5 --train_shots 5 --train_queries 1 --validation_split 0.0
# python3 train.py --label "20-WAY_SEED-17"  --seed 17 --train_classes 20 --train_ways 5 --train_shots 5 --train_queries 1 --validation_split 0.0
# python3 train.py --label "22-WAY_SEED-18"  --seed 18 --train_classes 22 --train_ways 5 --train_shots 5 --train_queries 1 --validation_split 0.0
# python3 train.py --label "24-WAY_SEED-19"  --seed 19 --train_classes 24 --train_ways 5 --train_shots 5 --train_queries 1 --validation_split 0.0
# python3 train.py --label "26-WAY_SEED-20"  --seed 20 --train_classes 26 --train_ways 5 --train_shots 5 --train_queries 1 --validation_split 0.0


SEED=10

for WAYS in $(seq 6 2 26)
do
  LABEL="${WAYS}-WAY_SEED-${SEED}"
  echo "Running training with: --label \"$LABEL\" --seed $SEED --train_classes $WAYS"
  python3 train.py \
    --label "$LABEL" \
    --seed "$SEED" \
    --train_classes "$WAYS" \
    --train_ways 5 \
    --train_shots 5 \
    --train_queries 1 \
    --validation_split 0.0
  SEED=$((SEED + 1))
done

echo "All training runs completed."
