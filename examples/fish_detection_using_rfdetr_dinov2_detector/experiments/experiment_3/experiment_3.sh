#!/bin/bash
###############################################################################
# experiment_3.sh — Full model fine-tuning: RF-DETR Nano + DINOv2
#
# Trains ALL parameters (backbone + decoder + head) with differential
# learning rates — the canonical RF-DETR training strategy.
#
# Strategy (aligned with RF-DETR defaults):
#   - train_mode = full          (nothing frozen)
#   - Backbone LR = 1.5e-4      (RF-DETR default lr_encoder)
#   - Decoder  LR = 7e-5        (head_lr × lr_component_decay = 1e-4 × 0.7)
#   - Head     LR = 1e-4        (RF-DETR default)
#   - ViT layer decay = 0.8     (deeper backbone layers get less LR)
#   - group_detr = 13           (RF-DETR default; matches pretrained query embeddings)
#   - clip_max_norm = 0.1       (RF-DETR default)
#   - warmup = 2 epochs         (brief ramp-up; backbone starts from pretrained)
#   - cosine LR schedule        (smooth decay)
#   - EMA decay = 0.993         (RF-DETR default for fine-tuning)
#
# This is the RECOMMENDED approach: the original RF-DETR trains everything
# from epoch 0 with differential LRs, not staged unfreezing.
#
# Model: RFDETRNano (DINOv2-small backbone, 384×384, 2 decoder layers)
# Dataset: DeepFish — 6,517 images, 1 class ("Fish"), ~3.7 annotations/image
#          80/20 split → ~5,214 train / ~1,303 val
#
# Usage:
#   sbatch experiment_3.sh
#   RESUME=1 sbatch experiment_3.sh
#
###############################################################################

#SBATCH --job-name=rfdetr_exp3_full
#SBATCH --partition=gpu_ampere
#SBATCH --account=deepl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=30-00:00:00
#SBATCH --chdir=/mnt/beegfs/home/mebrahim/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector
#SBATCH --output=/mnt/beegfs/home/mebrahim/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments/experiment_3/slurm_%j.out
#SBATCH --error=/mnt/beegfs/home/mebrahim/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments/experiment_3/slurm_%j.err

set -euo pipefail

###############################################################################
# Pre-create experiment directory (MUST exist before SLURM writes logs)
###############################################################################
EXP_BASE="/mnt/beegfs/home/mebrahim/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments/experiment_3"
mkdir -p "${EXP_BASE}/checkpoints" "${EXP_BASE}/plots"

###############################################################################
# XLA / cuDNN flags (Ampere safe-mode — avoid autotuner hangs)
###############################################################################
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"

###############################################################################
# Configurable parameters
###############################################################################
AUGMENTATION="${AUGMENTATION:-pipeline2}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-4}"
LR_ENCODER="${LR_ENCODER:-1.5e-4}"
LR_COMPONENT_DECAY="${LR_COMPONENT_DECAY:-0.7}"
LR_VIT_LAYER_DECAY="${LR_VIT_LAYER_DECAY:-0.8}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-2.0}"
CLIP_MAX_NORM="${CLIP_MAX_NORM:-0.1}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-20}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.3}"
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
EXP_DIR="${EXPERIMENTS_ROOT}/experiment_3"

###############################################################################
# Log experiment configuration
###############################################################################
echo "============================================================"
echo "  EXPERIMENT 3: Full model fine-tuning — RF-DETR Nano"
echo "============================================================"
echo "  Date           : $(date)"
echo "  Node           : $(hostname)"
echo "  GPU            : ${CUDA_VISIBLE_DEVICES:-none}"
echo "  Job ID         : ${SLURM_JOB_ID:-local}"
echo "  Partition      : ${SLURM_JOB_PARTITION:-interactive}"
echo "------------------------------------------------------------"
echo "  Variant        : RFDETRNano"
echo "  Train mode     : full"
echo "  Augmentation   : ${AUGMENTATION}"
echo "  Epochs         : ${EPOCHS}"
echo "  Batch size     : ${BATCH_SIZE}"
echo "  Head LR        : ${LR}"
echo "  Encoder LR     : ${LR_ENCODER}"
echo "  LR comp. decay : ${LR_COMPONENT_DECAY}"
echo "  ViT layer decay: ${LR_VIT_LAYER_DECAY}"
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
    "experiment": "experiment_3",
    "description": "Full model fine-tuning (RF-DETR defaults): RF-DETR Nano + DINOv2",
    "variant": "RFDETRNano",
    "train_mode": "full",
    "augmentation": "${AUGMENTATION}",
    "epochs": ${EPOCHS},
    "batch_size": ${BATCH_SIZE},
    "lr": ${LR},
    "lr_encoder": ${LR_ENCODER},
    "lr_component_decay": ${LR_COMPONENT_DECAY},
    "lr_vit_layer_decay": ${LR_VIT_LAYER_DECAY},
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
    "ema_decay": 0.993,
    "ema_tau": 100.0,
    "val_split": 0.2,
    "iou_threshold": 0.5,
    "group_detr": 13,
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
    --experiment-name "experiment_3" \
    --experiments-root "${EXPERIMENTS_ROOT}" \
    --variant RFDETRNano \
    --train-mode full \
    --augmentation "${AUGMENTATION}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --lr-encoder "${LR_ENCODER}" \
    --lr-component-decay "${LR_COMPONENT_DECAY}" \
    --lr-vit-layer-decay "${LR_VIT_LAYER_DECAY}" \
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
    --ema-decay 0.993 \
    --ema-tau 100.0 \
    --use-ema \
    --validate \
    ${RESUME_FLAG}

EXIT_CODE=$?

###############################################################################
# Post-training summary
###############################################################################
echo ""
echo "============================================================"
echo "  EXPERIMENT 3 COMPLETE — exit code: ${EXIT_CODE}"
echo "  Date           : $(date)"
echo "  Checkpoints    : ${EXP_DIR}/checkpoints/"
echo "  Plots          : ${EXP_DIR}/plots/"
echo "  Metrics log    : ${EXP_DIR}/metrics_log.json"
echo "  Config         : ${EXP_DIR}/experiment_config.json"
echo "============================================================"

exit ${EXIT_CODE}
