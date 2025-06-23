#!/bin/bash

# 고정할 하이퍼파라미터
GPU_NUM=1
HOP=1
INTERVAL_MINUTES=30
BATCH_SIZE=300

for SEED in 28 2025 1024 42 256; do
    echo ">>> Running with SEED=$SEED, BATCH_SIZE=$BATCH_SIZE, GPU_NUM=$GPU_NUM, HOP=$HOP, INTERVAL_MINUTES=$INTERVAL_MINUTES"
    
    env \
    SEED=$SEED \
    GPU_NUM=$GPU_NUM \
    INTERVAL_MINUTES=$INTERVAL_MINUTES \
    BATCH_SIZE=$BATCH_SIZE \
    HOP=$HOP \
    python Rec-with-TKG/model/main_3w.py
done
