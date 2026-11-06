#!/bin/bash
###############################################################################
# experiment_5.sh — RF-DETR Nano fine-tuning via the official high-level API
#
# Uses RFDETRNano.train() — the official RF-DETR training interface.
# All training logic (optimizer, LR schedule, EMA, loss, matching) is
# handled internally by the framework.  The Python script applies two
# targeted monkey-patches to work around Keras-port bugs:
#   1. engine.train_one_epoch Phase 1 uses training=True (not False)
#   2. Warm the training-mode JAX trace after model init
#   3. Inject cosine LR schedule (base_lr=1e-4) into AdamW optimizer
#
# Strategy (RF-DETR defaults):
#   - Full model fine-tuning (RF-DETR default, no freeze flags)
#   - Base LR = 1e-4 (cosine schedule → 0)
#   - Backbone LR = 1.5e-4      (lr_encoder, dead in train_from_config)
#   - Decoder  LR = 7e-5        (lr × lr_component_decay, dead)
#   - LR schedule: cosine annealing (via monkey-patched AdamW)
#   - ViT layer decay = 0.8     (dead in train_from_config)
#   - group_detr = 13
#   - clip_max_norm = 0.1
#   - warmup = 0 epochs
#   - EMA decay = 0.993
#   - Data: DetectionDataGenerator (pipeline2 augmentation + ImageNet norm)
#   - Per-step progress tee'd to output.txt
#
# Model: RFDETRNano (DINOv2-small backbone, 384×384, 2 decoder layers)
# Dataset: DeepFish — 6,517 images, 1 class ("Fish"), ~3.7 annotations/image
#          80/20 split → ~5,214 train / ~1,303 val
#
# Usage:
#   sbatch experiment_5.sh
#
###############################################################################

#SBATCH --job-name=rfdetr_exp4_hlapi
#SBATCH --partition=gpu_ampere
#SBATCH --account=deepl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=30-00:00:00
#SBATCH --chdir=/mnt/beegfs/home/mebrahim/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector
#SBATCH --output=/mnt/beegfs/home/mebrahim/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments/experiment_5/slurm_%j.out
#SBATCH --error=/mnt/beegfs/home/mebrahim/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments/experiment_5/slurm_%j.err

set -euo pipefail

###############################################################################
# Pre-create experiment directory (MUST exist before SLURM writes logs)
###############################################################################
EXP_BASE="/mnt/beegfs/home/mebrahim/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments/experiment_5"
mkdir -p "${EXP_BASE}/checkpoints" "${EXP_BASE}/plots"

###############################################################################
# XLA / cuDNN flags (Ampere safe-mode — avoid autotuner hangs)
###############################################################################
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"

###############################################################################
# Paths
###############################################################################
CONDA_ENV="/mnt/beegfs/home/mebrahim/miniconda3/envs/paz_jax_dev_environment"
PYTHON="${CONDA_ENV}/bin/python"
SCRIPT_DIR="/mnt/beegfs/home/mebrahim/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector"

###############################################################################
# Log experiment configuration
###############################################################################
echo "============================================================"
echo "  EXPERIMENT 5: RF-DETR High-Level API — RF-DETR Nano"
echo "============================================================"
echo "  Date           : $(date)"
echo "  Node           : $(hostname)"
echo "  GPU            : ${CUDA_VISIBLE_DEVICES:-none}"
echo "  Job ID         : ${SLURM_JOB_ID:-local}"
echo "  Partition      : ${SLURM_JOB_PARTITION:-interactive}"
echo "------------------------------------------------------------"
echo "  Variant        : RFDETRNano"
echo "  API            : RFDETRNano.train() (high-level)"
echo "  Epochs         : 20"
echo "  Batch size     : 16"
echo "  Base LR        : 1e-4"
echo "  LR schedule    : cosine → 0"
echo "  Encoder LR     : 1.5e-4 (config only, not used by train_from_config)"
echo "  LR comp. decay : 0.7 (config only, not used by train_from_config)"
echo "  ViT layer decay: 0.8 (config only, not used by train_from_config)"
echo "  Weight decay   : 1e-4"
echo "  Warmup epochs  : 0.0"
echo "  Clip max norm  : 0.1"
echo "  EMA decay      : 0.993"
echo "  group_detr     : 13"
echo "  Early stopping : yes"
echo "  Output dir     : ${EXP_BASE}"
echo "============================================================"

# Save config to JSON for reproducibility
cat > "${EXP_BASE}/experiment_config.json" <<EOF
{
    "experiment": "experiment_5",
    "description": "RF-DETR high-level API fine-tuning: RFDETRNano.train() on DeepFish",
    "variant": "RFDETRNano",
    "api": "high-level (RFDETRNano.train)",
    "epochs": 20,
    "batch_size": 16,
    "lr": 1e-4,
    "lr_schedule": "cosine",
    "lr_min_factor": 0.0,
    "lr_encoder": 1.5e-4,
    "lr_component_decay": 0.7,
    "lr_vit_layer_decay": 0.8,
    "weight_decay": 1e-4,
    "warmup_epochs": 0.0,
    "clip_max_norm": 0.1,
    "ema_decay": 0.993,
    "ema_tau": 100,
    "group_detr": 13,
    "early_stopping": true,
    "early_stopping_patience": 10,
    "checkpoint_interval": 10,
    "val_split": 0.2,
    "seed": 42,
    "dataset": "DeepFish (6517 images, 1 class)",
    "normalization": "ImageNet (pipeline2 augmentation + ImageNet norm via _AugmentedDataLoader)",
    "validation": "validate_epoch_full per epoch (val + train-eval)",
    "checkpointing": "best_keep strategy → checkpoints/rfdetr_nano_epoch_EEEE_val_loss_V.VVVV_mAP_M.MMMM.weights.h5",
    "plots": "MetricsTracker plots every epoch → plots/",
    "monkey_patches": [
        "engine.train_one_epoch: training=True in Phase 1 eager forward",
        "warm training-mode JAX trace after model init",
        "AdamW constructor: inject cosine LambdaLRSchedule (base_lr=1e-4)"
    ],
    "date": "$(date -Iseconds)",
    "node": "$(hostname)",
    "job_id": "${SLURM_JOB_ID:-local}"
}
EOF

###############################################################################
# Launch training
###############################################################################
${PYTHON} ${SCRIPT_DIR}/experiments/experiment_5/experiment_5.py

EXIT_CODE=$?

###############################################################################
# Post-training summary
###############################################################################
echo ""
echo "============================================================"
echo "  EXPERIMENT 5 COMPLETE — exit code: ${EXIT_CODE}"
echo "  Date           : $(date)"
echo "  Metrics log    : ${EXP_BASE}/metrics_log.json"
echo "  Best checkpoint: ${EXP_BASE}/checkpoints/rfdetr_nano_best.weights.h5"
echo "  Final weights  : ${EXP_BASE}/checkpoints/rfdetr_nano_final.weights.h5"
echo "  Config         : ${EXP_BASE}/experiment_config.json"
echo "============================================================"

exit ${EXIT_CODE}
