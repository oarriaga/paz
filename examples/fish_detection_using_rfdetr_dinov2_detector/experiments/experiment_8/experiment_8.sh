#!/bin/bash
###############################################################################
# experiment_8.sh — RF-DETR Nano fast fixed-shape benchmark
#
# Speed-focused variant of Experiment 7 that preserves the same high-level API
# and monitoring style while removing the largest known step-time multipliers.
#
# Key changes relative to Experiment 7:
#   - grad_accum_steps = 1
#   - multi_scale = no
#   - fixed 384x384 training shape
#   - thread-based native prefetch enabled (num_workers=2)
#
# Usage:
#   sbatch experiment_8.sh
#
###############################################################################

#SBATCH --job-name=rfdetr_exp8_fast
#SBATCH --partition=gpu_ampere
#SBATCH --account=deepl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=30-00:00:00
#SBATCH --chdir=/mnt/beegfs/home/mebrahim/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector
#SBATCH --output=/mnt/beegfs/home/mebrahim/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments/experiment_8/slurm_%j.out
#SBATCH --error=/mnt/beegfs/home/mebrahim/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments/experiment_8/slurm_%j.err

set -euo pipefail

EXP_BASE="/mnt/beegfs/home/mebrahim/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments/experiment_8"
mkdir -p "${EXP_BASE}/checkpoints" "${EXP_BASE}/plots"

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"

CONDA_ENV="/mnt/beegfs/home/mebrahim/miniconda3/envs/paz_jax_dev_environment"
PYTHON="${CONDA_ENV}/bin/python"
SCRIPT_DIR="/mnt/beegfs/home/mebrahim/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector"

echo "============================================================"
echo "  EXPERIMENT 8: RF-DETR Nano — Fast Fixed-Shape Benchmark"
echo "============================================================"
echo "  Date           : $(date)"
echo "  Node           : $(hostname)"
echo "  GPU            : ${CUDA_VISIBLE_DEVICES:-none}"
echo "  Job ID         : ${SLURM_JOB_ID:-local}"
echo "  Partition      : ${SLURM_JOB_PARTITION:-interactive}"
echo "------------------------------------------------------------"
echo "  Variant        : RFDETRNano"
echo "  API            : train_from_config (high-level)"
echo "  Epochs         : 40"
echo "  Batch size     : 4"
echo "  Grad accum     : 1 (effective batch = 4)"
echo "  Base LR        : 1e-4"
echo "  Encoder LR     : 1.5e-4"
echo "  LR schedule    : cosine (warmup=2, min_factor=0.01)"
echo "  LR comp. decay : 0.7"
echo "  ViT layer decay: 0.8"
echo "  Weight decay   : 1e-4"
echo "  Clip max norm  : 0.1"
echo "  EMA decay      : 0.993 (tau=100)"
echo "  Drop path      : 0.1"
echo "  Fixed shape    : 384x384"
echo "  Multi-scale    : no"
echo "  Prefetch       : yes (thread-based, num_workers=2)"
echo "  AMP (bfloat16) : yes"
echo "  Evaluation     : pycocotools COCO AP"
echo "  Early stopping : patience=15, delta=0.0005"
echo "  Output dir     : ${EXP_BASE}"
echo "============================================================"

cat > "${EXP_BASE}/experiment_config_shell.json" <<EOF
{
    "experiment": "experiment_8",
    "variant": "RFDETRNano",
    "resolution": 384,
    "fixed_training_shape": 384,
    "epochs": 40,
    "batch_size": 4,
    "grad_accum_steps": 1,
    "effective_batch_size": 4,
    "lr": 1e-4,
    "lr_encoder": 1.5e-4,
    "lr_scheduler": "cosine",
    "lr_min_factor": 0.01,
    "lr_component_decay": 0.7,
    "lr_vit_layer_decay": 0.8,
    "warmup_epochs": 2.0,
    "weight_decay": 1e-4,
    "clip_max_norm": 0.1,
    "ema_decay": 0.993,
    "ema_tau": 100,
    "drop_path": 0.1,
    "multi_scale": false,
    "expanded_scales": false,
    "num_workers": 2,
    "amp": true,
    "evaluation": "pycocotools COCO AP",
    "dataset": "DeepFish",
    "early_stopping_patience": 15,
    "early_stopping_min_delta": 0.0005,
    "group_detr": 13
}
EOF

${PYTHON} ${SCRIPT_DIR}/experiments/experiment_8/experiment_8.py

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "  EXPERIMENT 8 COMPLETE — exit code: ${EXIT_CODE}"
echo "  Date           : $(date)"
echo "  Metrics log    : ${EXP_BASE}/metrics_log.json"
echo "  Best ckpt      : ${EXP_BASE}/checkpoints/rfdetr_nano_best.weights.h5"
echo "  Final weights  : ${EXP_BASE}/checkpoints/rfdetr_nano_final.weights.h5"
echo "  Config         : ${EXP_BASE}/experiment_config.json"
echo "============================================================"

exit ${EXIT_CODE}