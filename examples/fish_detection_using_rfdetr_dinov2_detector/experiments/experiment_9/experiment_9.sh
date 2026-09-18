#!/bin/bash
###############################################################################
# experiment_9.sh — RF-DETR Nano with native RF-DETR augmentation
#
# Based on Experiment 5 with one change: the training data pipeline uses
# the native RF-DETR augmentation (make_coco_transforms_square_div_64)
# instead of the custom pipeline2.
#
# Native RF-DETR train augmentation (multi_scale=False, square_div_64=True):
#   - RandomHorizontalFlip  (p = 0.5)
#   - RandomSelect(
#         SquareResize([384]),
#         Compose([RandomResize([400,500,600]), RandomSizeCrop(384,600),
#                  SquareResize([384])])
#     )
#   - ToTensor + Normalize(ImageNet mean/std)
#
# All hyperparameters are identical to Experiment 5:
#   batch_size=16, epochs=20, cosine LR 1e-4→0, warmup=0, EMA=0.993
#
# The critical engine patch (training=True in Phase 1 of train_one_epoch)
# is still applied — same as Experiment 5.
#
# Usage:
#   sbatch experiment_9.sh
#
###############################################################################

#SBATCH --job-name=rfdetr_exp9_native_aug
#SBATCH --partition=gpu_ampere
#SBATCH --account=deepl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=30-00:00:00
#SBATCH --chdir=/mnt/beegfs/home/mebrahim/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector
#SBATCH --output=/mnt/beegfs/home/mebrahim/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments/experiment_9/slurm_%j.out
#SBATCH --error=/mnt/beegfs/home/mebrahim/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments/experiment_9/slurm_%j.err

set -euo pipefail

EXP_BASE="/mnt/beegfs/home/mebrahim/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector/experiments/experiment_9"
mkdir -p "${EXP_BASE}/checkpoints" "${EXP_BASE}/plots"

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"

CONDA_ENV="/mnt/beegfs/home/mebrahim/miniconda3/envs/paz_jax_dev_environment"
PYTHON="${CONDA_ENV}/bin/python"
SCRIPT_DIR="/mnt/beegfs/home/mebrahim/projects/fish_detector_using_rfdetr/paz/examples/fish_detection_using_rfdetr_dinov2_detector"

echo "============================================================"
echo "  EXPERIMENT 9: RF-DETR Nano — Native RF-DETR Augmentation"
echo "============================================================"
echo "  Date           : $(date)"
echo "  Node           : $(hostname)"
echo "  GPU            : ${CUDA_VISIBLE_DEVICES:-none}"
echo "  Job ID         : ${SLURM_JOB_ID:-local}"
echo "  Partition      : ${SLURM_JOB_PARTITION:-interactive}"
echo "------------------------------------------------------------"
echo "  Variant        : RFDETRNano"
echo "  API            : RFDETRNano.train_from_config (high-level)"
echo "  Augmentation   : native RF-DETR (RandomHorizontalFlip +"
echo "                   RandomSelect(SquareResize | RandomSizeCrop+"
echo "                   SquareResize)) + ImageNet norm"
echo "  Epochs         : 100"
echo "  Batch size     : 16"
echo "  Grad accum     : 1 (effective batch = 16)"
echo "  Base LR        : 1e-4"
echo "  LR schedule    : cosine -> 0 (warmup=0)"
echo "  Encoder LR     : 1.5e-4"
echo "  LR comp. decay : 0.7"
echo "  ViT layer decay: 0.8"
echo "  Weight decay   : 1e-4"
echo "  Clip max norm  : 0.1"
echo "  EMA decay      : 0.993 (tau=100)"
echo "  group_detr     : 13"
echo "  drop_path      : 0.0"
echo "  Multi-scale    : no (fixed 384x384)"
echo "  square_div_64  : yes"
echo "  AMP (bfloat16) : yes"
echo "  Early stopping : patience=10, delta=0.001"
echo "  Engine patch   : training=True in Phase 1 (group_detr fix)"
echo "  Output dir     : ${EXP_BASE}"
echo "============================================================"

${PYTHON} "${SCRIPT_DIR}/experiments/experiment_9/experiment_9.py"

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "  EXPERIMENT 9 COMPLETE — exit code: ${EXIT_CODE}"
echo "  Date           : $(date)"
echo "  Metrics log    : ${EXP_BASE}/metrics_log.json"
echo "  Best checkpoint: ${EXP_BASE}/checkpoints/rfdetr_nano_best.weights.h5"
echo "  Final weights  : ${EXP_BASE}/checkpoints/rfdetr_nano_final.weights.h5"
echo "  Config         : ${EXP_BASE}/experiment_config.json"
echo "============================================================"

exit ${EXIT_CODE}
