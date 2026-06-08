#!/bin/bash

mkdir -p runs

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="runs/distributed_run_$TIMESTAMP.log"

GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")

echo "======================================" | tee $LOG_FILE
echo "Distributed Training Launcher" | tee -a $LOG_FILE
echo "Detected GPUs: $GPU_COUNT" | tee -a $LOG_FILE
echo "Arguments: $@" | tee -a $LOG_FILE
echo "======================================" | tee -a $LOG_FILE

if [[ "$@" == *"--tracker wandb"* ]]; then
    if [ -z "$WANDB_API_KEY" ]; then
        echo "ERROR: No WandB API key found."
        echo "Please login first:"
        echo "    wandb login YOUR_API_KEY"
        exit 1
    fi
fi

if [ "$GPU_COUNT" -ge 2 ]; then
    echo "Launching multi-GPU training with torchrun..." | tee -a $LOG_FILE
    torchrun --nproc_per_node=$GPU_COUNT train.py "$@" 2>&1 | tee -a $LOG_FILE
else
    echo "Only $GPU_COUNT GPU detected. Falling back to single-process training..." | tee -a $LOG_FILE
    python train.py "$@" 2>&1 | tee -a $LOG_FILE
fi
