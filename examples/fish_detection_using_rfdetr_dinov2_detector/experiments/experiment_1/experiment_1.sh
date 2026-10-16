#!/bin/bash
###############################################################################
# experiment_1.sh — Full-dataset head-only training: RF-DETR Nano + DINOv2
#
# Trains only the detection head (class_embed + bbox_embed MLPs and query
# embeddings) while freezing the DINOv2 backbone and transformer decoder.
#
# Model: RFDETRNano (DINOv2-small backbone, 384×384, 2 decoder layers)
# Dataset: DeepFish — 6,517 images, 1 class ("Fish"), ~3.7 annotations/image
#          80/20 split → ~5,214 train / ~1,303 val
#
# Augmentation is configurable via the AUGMENTATION environment variable:
#   AUGMENTATION=pipeline2   (default — horizontal flip + color jitter)
#   AUGMENTATION=rf_detr     (reserved for future RF-DETR native augmentations)
#
# Usage:
#   sbatch experiment_1.sh                          # defaults
#   AUGMENTATION=rf_detr sbatch experiment_1.sh     # override augmentation
#   EPOCHS=100 sbatch experiment_1.sh               # override epochs
#   RESUME=1 sbatch experiment_1.sh                 # resume from checkpoint
#
###############################################################################

#SBATCH --job-name=rfdetr_exp1_head
#SBATCH --partition=gpu_ampere
#SBATCH --account=deepl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --chdir=/mnt/beegfs/home/mebrahim/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector
#SBATCH --output=/mnt/beegfs/home/mebrahim/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments/experiment_1/slurm_%j.out
#SBATCH --error=/mnt/beegfs/home/mebrahim/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments/experiment_1/slurm_%j.err

set -euo pipefail

###############################################################################
# Pre-create experiment directory (MUST exist before SLURM writes logs)
###############################################################################
EXP_BASE="/mnt/beegfs/home/mebrahim/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments/experiment_1"
mkdir -p "${EXP_BASE}/checkpoints" "${EXP_BASE}/plots"

###############################################################################
# XLA / cuDNN flags (Ampere safe-mode — avoid autotuner hangs)
###############################################################################
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"

###############################################################################
# Configurable parameters (override via environment before sbatch)
###############################################################################
AUGMENTATION="${AUGMENTATION:-pipeline2}"     # pipeline2 | rf_detr
EPOCHS="${EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-1.0}"
CLIP_MAX_NORM="${CLIP_MAX_NORM:-1.0}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-20}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_SIZE="${PREFETCH_SIZE:-8}"
SEED="${SEED:-42}"

# Resume flag
RESUME_FLAG=""
if [[ "${RESUME:-0}" == "1" ]]; then
    RESUME_FLAG="--resume"
fi

###############################################################################
# Paths
###############################################################################
CONDA_ENV="/mnt/beegfs/home/mebrahim/miniconda3/envs/paz_jax_dev_environment"
PYTHON="${CONDA_ENV}/bin/python"
SCRIPT_DIR="/mnt/beegfs/home/mebrahim/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector"
TRAIN_SCRIPT="${SCRIPT_DIR}/src/train.py"
EXPERIMENTS_ROOT="${SCRIPT_DIR}/experiments"
EXP_DIR="${EXPERIMENTS_ROOT}/experiment_1"

# (directories already created at top of script)

###############################################################################
# Log experiment configuration
###############################################################################
echo "============================================================"
echo "  EXPERIMENT 1: Full-dataset head-only — RF-DETR Nano"
echo "============================================================"
echo "  Date           : $(date)"
echo "  Node           : $(hostname)"
echo "  GPU            : ${CUDA_VISIBLE_DEVICES:-none}"
echo "  Job ID         : ${SLURM_JOB_ID:-local}"
echo "  Partition      : ${SLURM_JOB_PARTITION:-interactive}"
echo "------------------------------------------------------------"
echo "  Variant        : RFDETRNano"
echo "  Train mode     : head_only"
echo "  Augmentation   : ${AUGMENTATION}"
echo "  Epochs         : ${EPOCHS}"
echo "  Batch size     : ${BATCH_SIZE}"
echo "  Learning rate  : ${LR}"
echo "  Weight decay   : ${WEIGHT_DECAY}"
echo "  Warmup epochs  : ${WARMUP_EPOCHS}"
echo "  Clip max norm  : ${CLIP_MAX_NORM}"
echo "  Conf. threshold: ${CONFIDENCE_THRESHOLD}"
echo "  Early stop pat.: ${EARLY_STOPPING_PATIENCE}"
echo "  Num workers    : ${NUM_WORKERS}"
echo "  Prefetch size  : ${PREFETCH_SIZE}"
echo "  Seed           : ${SEED}"
echo "  Resume         : ${RESUME:-0}"
echo "  Output dir     : ${EXP_DIR}"
echo "============================================================"

# Save config to JSON for reproducibility
cat > "${EXP_DIR}/experiment_config.json" <<EOF
{
    "experiment": "experiment_1",
    "description": "Full-dataset head-only training: RF-DETR Nano + DINOv2",
    "variant": "RFDETRNano",
    "train_mode": "head_only",
    "augmentation": "${AUGMENTATION}",
    "epochs": ${EPOCHS},
    "batch_size": ${BATCH_SIZE},
    "lr": ${LR},
    "weight_decay": ${WEIGHT_DECAY},
    "warmup_epochs": ${WARMUP_EPOCHS},
    "clip_max_norm": ${CLIP_MAX_NORM},
    "confidence_threshold": ${CONFIDENCE_THRESHOLD},
    "early_stopping_patience": ${EARLY_STOPPING_PATIENCE},
    "num_workers": ${NUM_WORKERS},
    "prefetch_size": ${PREFETCH_SIZE},
    "seed": ${SEED},
    "lr_scheduler": "cosine",
    "lr_min_factor": 0.01,
    "ema_decay": 0.9997,
    "ema_tau": 2000.0,
    "val_split": 0.2,
    "iou_threshold": 0.5,
    "dataset": "DeepFish (6517 images, 1 class)",
    "date": "$(date -Iseconds)",
    "node": "$(hostname)",
    "job_id": "${SLURM_JOB_ID:-local}"
}
EOF

###############################################################################
# Launch training
###############################################################################
${PYTHON} ${TRAIN_SCRIPT} \
    --experiment-name "experiment_1" \
    --experiments-root "${EXPERIMENTS_ROOT}" \
    --variant RFDETRNano \
    --train-mode head_only \
    --augmentation "${AUGMENTATION}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --lr-encoder 0.0 \
    --lr-component-decay 0.0 \
    --warmup-epochs "${WARMUP_EPOCHS}" \
    --lr-scheduler cosine \
    --lr-min-factor 0.01 \
    --weight-decay "${WEIGHT_DECAY}" \
    --clip-max-norm "${CLIP_MAX_NORM}" \
    --early-stopping \
    --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
    --early-stopping-min-delta 1e-4 \
    --checkpoint-mode best_keep \
    --plot-interval 1 \
    --seed "${SEED}" \
    --val-split 0.2 \
    --confidence-threshold "${CONFIDENCE_THRESHOLD}" \
    --iou-threshold 0.5 \
    --group-detr 13 \
    --print-freq 10 \
    --num-workers "${NUM_WORKERS}" \
    --prefetch-size "${PREFETCH_SIZE}" \
    --ema-decay 0.9997 \
    --ema-tau 2000.0 \
    --use-ema \
    --validate \
    ${RESUME_FLAG}

EXIT_CODE=$?

###############################################################################
# Post-training summary
###############################################################################
echo ""
echo "============================================================"
echo "  EXPERIMENT 1 COMPLETE — exit code: ${EXIT_CODE}"
echo "  Date           : $(date)"
echo "  Checkpoints    : ${EXP_DIR}/checkpoints/"
echo "  Plots          : ${EXP_DIR}/plots/"
echo "  Metrics log    : ${EXP_DIR}/metrics_log.json"
echo "  Config         : ${EXP_DIR}/experiment_config.json"
echo "============================================================"

exit ${EXIT_CODE}
