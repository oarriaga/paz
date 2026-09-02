import os
import sys
import argparse
import random
import time
import math
import datetime

import numpy as np

# ---------------------------------------------------------------------------
# Environment (set before any framework import)
# ---------------------------------------------------------------------------
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# ---------------------------------------------------------------------------
# Ensure the paz package is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PAZ_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _PAZ_ROOT not in sys.path:
    sys.path.insert(0, _PAZ_ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# ---------------------------------------------------------------------------
# Imports — project modules
# ---------------------------------------------------------------------------
from train_utils import (
    VARIANT_MAP,
    VARIANT_FRIENDLY_NAME,
    _DEFAULT_EXPERIMENTS_ROOT,
    prepare_coco_dataset,
    validate_epoch_full,
    setup_logging,
)
from training_helpers import (
    count_parameters,
    count_component_parameters,
    log_model_summary,
    apply_train_mode,
    verify_frozen_gradients,
    EarlyStopper,
    build_lr_schedule,
    build_param_group_schedules,
    compute_gradient_norm,
    make_batches,
    log_default_config,
    train_one_epoch_custom,
)
from dataset import DeepFishDataset, DEEPFISH_CLASS_NAMES
from metrics_tracker import MetricsTracker
from generator import DetectionDataGenerator, prefetch_iterator


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser():
    """Build the CLI argument parser for all training options.

    Returns
    -------
    argparse.ArgumentParser
    """
    p = argparse.ArgumentParser(
        description="RF-DETR fish training — research-grade pipeline (v3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Experiment -------------------------------------------------------
    grp = p.add_mutually_exclusive_group()
    grp.add_argument(
        "--experiment-name", type=str, default=None,
        help="Experiment folder name (experiments/<name>/)",
    )
    grp.add_argument(
        "--experiment-number", type=int, default=None,
        help="Experiment number N -> experiments/experiment_N/",
    )
    p.add_argument(
        "--experiments-root", type=str, default=_DEFAULT_EXPERIMENTS_ROOT,
        help="Root directory for all experiment outputs",
    )

    # -- Data -------------------------------------------------------------
    p.add_argument(
        "--deepfish-root", type=str, default=None,
        help="Root dir for DeepFish dataset (default: ~/.keras/paz/datasets/Deepfish)",
    )

    # -- Model ------------------------------------------------------------
    p.add_argument(
        "--variant", type=str, default="RFDETRSmall",
        choices=list(VARIANT_MAP.keys()),
        help="RF-DETR model variant",
    )
    p.add_argument(
        "--weights", type=str, default="default",
        help="'default' (pretrained COCO), 'none' (random init), "
             "or a path to a .weights.h5 file",
    )

    # -- Training hyper-parameters ----------------------------------------
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-encoder", type=float, default=1.5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--clip-max-norm", type=float, default=0.1)
    p.add_argument("--warmup-epochs", type=float, default=5.0,
                   help="Number of warmup epochs before main LR schedule")
    p.add_argument("--lr-vit-layer-decay", type=float, default=0.8)
    p.add_argument("--lr-component-decay", type=float, default=0.7,
                   help="LR multiplier for decoder relative to head LR")
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--lr-drop", type=int, default=100,
                   help="Epoch at which LR drops (for step/multistep)")
    p.add_argument(
        "--group-detr", type=int, default=13,
        help="Number of query groups for group-DETR matching. "
             "Default 13 (RF-DETR default). Must match pretrained "
             "weights for refpoint_embed / query_feat to load.",
    )

    # -- LR Scheduler -----------------------------------------------------
    p.add_argument(
        "--lr-scheduler", type=str, default="cosine",
        choices=["cosine", "step", "multistep", "one_cycle"],
        help="Learning rate scheduler type",
    )
    p.add_argument(
        "--lr-milestones", type=int, nargs="+", default=None,
        help="Epoch milestones for multistep scheduler",
    )
    p.add_argument(
        "--lr-gamma", type=float, default=0.1,
        help="LR multiplicative factor for step/multistep",
    )
    p.add_argument(
        "--lr-min-factor", type=float, default=0.01,
        help="Minimum LR as fraction of peak (for cosine/one_cycle)",
    )

    # -- Train Mode -------------------------------------------------------
    p.add_argument(
        "--train-mode", type=str, default="full",
        choices=["full", "decoder_only", "head_only"],
        help="Training mode: 'full' (all params), 'decoder_only' "
             "(freeze backbone), 'head_only' (freeze backbone+decoder)",
    )

    # -- EMA --------------------------------------------------------------
    p.add_argument("--use-ema", action="store_true", default=True)
    p.add_argument("--no-ema", dest="use_ema", action="store_false")
    p.add_argument("--ema-decay", type=float, default=0.993)
    p.add_argument("--ema-tau", type=float, default=100.0)

    # -- Data splitting ---------------------------------------------------
    p.add_argument("--val-split", type=float, default=0.1,
                   help="Fraction of data for validation")
    p.add_argument("--subset", type=int, default=None,
                   help="Limit dataset to first N images (for quick tests)")
    p.add_argument("--seed", type=int, default=42)

    # -- Validation -------------------------------------------------------
    p.add_argument("--validate", action="store_true", default=True,
                   help="Run per-epoch validation")
    p.add_argument("--no-validate", dest="validate", action="store_false")
    p.add_argument(
        "--confidence-threshold", type=float, default=0.3,
        help="Confidence threshold for detection metrics",
    )
    p.add_argument(
        "--iou-threshold", type=float, default=0.5,
        help="IoU threshold for TP/FP matching in mAP computation",
    )
    p.add_argument(
        "--max-batches", type=int, default=None,
        help="Limit validation to this many batches per epoch",
    )

    # -- Checkpointing ----------------------------------------------------
    p.add_argument(
        "--checkpoint-mode", type=str, default="best_keep",
        choices=["best_keep", "best_replace", "every"],
        help="Checkpoint strategy",
    )

    # -- Plotting ---------------------------------------------------------
    p.add_argument("--plot-interval", type=int, default=1,
                   help="Generate plots every N epochs (1 = every epoch)")

    # -- Early stopping ---------------------------------------------------
    p.add_argument("--early-stopping", action="store_true", default=True)
    p.add_argument("--no-early-stopping", dest="early_stopping",
                   action="store_false")
    p.add_argument("--early-stopping-patience", type=int, default=10)
    p.add_argument("--early-stopping-min-delta", type=float, default=1e-4)

    # -- Augmentation -----------------------------------------------------
    p.add_argument(
        "--augmentation", type=str, default="pipeline2",
        choices=["none", "pipeline2", "rf_detr"],
        help="Augmentation strategy: 'none' (no augmentation), "
             "'pipeline2' (horizontal flip + color jitter), "
             "'rf_detr' (reserved for future RF-DETR native "
             "augmentations, currently no-op)",
    )

    # -- Resume -----------------------------------------------------------
    p.add_argument("--resume", action="store_true",
                   help="Resume from latest checkpoint in experiment dir")

    # -- Data loading -----------------------------------------------------
    p.add_argument("--num-workers", type=int, default=2,
                   help="Background threads for data loading (0=sync)")
    p.add_argument("--prefetch-size", type=int, default=4,
                   help="Number of batches to prefetch in background")

    # -- Logging ----------------------------------------------------------
    p.add_argument("--print-freq", type=int, default=10,
                   help="Print training stats every N steps")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = build_parser().parse_args()

    # ---- Seed -----------------------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import keras
        keras.utils.set_random_seed(args.seed)
    except Exception:
        pass

    # ---- Experiment directory -------------------------------------------
    if args.experiment_name:
        exp_dir = os.path.join(args.experiments_root, args.experiment_name)
    elif args.experiment_number is not None:
        exp_dir = os.path.join(
            args.experiments_root, f"experiment_{args.experiment_number}",
        )
    else:
        exp_dir = os.path.join(args.experiments_root, "experiment_1")

    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)

    log = setup_logging(exp_dir)
    log.info("=" * 68)
    log.info("RF-DETR Training v3 — %s — mode=%s",
             args.variant, args.train_mode)
    log.info("=" * 68)
    log.info("Experiment directory: %s", exp_dir)
    log.info("Config: %s", vars(args))
    log.info("Augmentation strategy: %s", args.augmentation)

    # ---- Log default configuration policy --------------------------------
    log_default_config(args, log)

    # ---- Dataset (metadata + COCO conversion) ---------------------------
    ds = DeepFishDataset(
        root=args.deepfish_root,
        subset=args.subset,
    )
    log.info("DeepFish dataset: %d images, %d classes",
             len(ds), ds.num_classes)
    log.info("Class names: %s", ds.class_names)

    coco_dir, train_indices, val_indices = prepare_coco_dataset(
        ds, exp_dir,
        val_split=args.val_split,
        seed=args.seed,
    )
    log.info("COCO data prepared: %d train, %d val -> %s",
             len(train_indices), len(val_indices), coco_dir)

    # ---- Instantiate model (via high-level RFDETR API) ------------------
    VariantClass = VARIANT_MAP[args.variant]

    weight_mode = args.weights.strip().lower()
    variant_kwargs = {"group_detr": args.group_detr}
    if weight_mode == "none":
        variant_kwargs["pretrain_weights"] = None
    elif weight_mode != "default":
        variant_kwargs["pretrain_weights"] = args.weights

    detector = VariantClass(**variant_kwargs)
    resolution = detector.model_config.resolution
    log.info("Model: %s  resolution=%d  pretrain=%s",
             args.variant, resolution,
             detector.model_config.pretrain_weights or "none")

    # ---- Reinitialise detection head for dataset classes -----------------
    if detector.model_config.num_classes != ds.num_classes:
        # Save pretrain weights path — reinitialize_detection_head rebuilds
        # the entire LWDETR from scratch, discarding ALL loaded weights.
        pretrain_file = detector.model_config.pretrain_weights

        detector.model.reinitialize_detection_head(ds.num_classes)
        detector.model_config = detector.model.config
        log.info("Detection head reinitialised for %d classes", ds.num_classes)

        # Reload pretrained weights into the new model.  skip_mismatch=True
        # restores backbone + transformer + bbox_embed (same shapes) and
        # skips class_embed / enc_out_class_embed (shape changed: %d → %d).
        if pretrain_file is not None:
            from paz.models.detection.dino_v2_object_detection.main import (
                resolve_weights_path,
            )
            wpath = resolve_weights_path(pretrain_file)
            if wpath is not None:
                # Build ALL layers (training=True) so every
                # enc_out_*[group_idx] head is constructed before
                # weight loading.  training=False only builds group 0.
                res = detector.model_config.resolution
                dummy = np.ones((1, res, res, 3), dtype="float32") * 0.5
                detector.model.model(dummy, training=True)
                detector.model.model.load_weights(wpath, skip_mismatch=True)
                log.info("Pretrained weights reloaded after head reinit "
                         "(skip_mismatch=True): %s", wpath)
            else:
                log.warning("Pretrained weights file '%s' not found — "
                            "model has RANDOM weights!", pretrain_file)

    keras_model = detector.model.model  # underlying Keras LWDETR

    # ---- Apply train mode (freezing) ------------------------------------
    freeze_info = apply_train_mode(keras_model, args.train_mode, logger=log)

    # ---- Pre-training diagnostic logging --------------------------------
    log_model_summary(
        keras_model, detector.model_config,
        args.batch_size, args.grad_accum_steps, log,
    )

    # ---- Verify frozen variable configuration ---------------------------
    verify_ok = verify_frozen_gradients(
        grads=None, model=keras_model, mode=args.train_mode, logger=log,
    )
    if not verify_ok:
        log.error("Frozen gradient verification FAILED before training!")

    # ---- Build criterion ------------------------------------------------
    from paz.models.detection.dino_v2_object_detection.main import (
        build_criterion_from_config,
    )
    criterion, postprocess = build_criterion_from_config(detector.model_config)

    # ---- Validation dataset (with model resolution) ---------------------
    val_criterion = None
    val_ds = None
    if args.validate:
        val_criterion, _ = build_criterion_from_config(detector.model_config)
        val_ds = DeepFishDataset(
            root=args.deepfish_root,
            resolution=resolution,
            subset=args.subset,
        )
        log.info("Validation enabled (every epoch, conf=%.2f, IoU=%.2f)",
                 args.confidence_threshold, args.iou_threshold)

    # ---- Training dataset (with model resolution) -----------------------
    train_ds = DeepFishDataset(
        root=args.deepfish_root,
        resolution=resolution,
        subset=args.subset,
    )

    # ---- MetricsTracker -------------------------------------------------
    model_name = VARIANT_FRIENDLY_NAME.get(args.variant, args.variant.lower())
    tracker = MetricsTracker(
        output_dir=exp_dir,
        model_name=model_name,
        plot_interval=args.plot_interval,
        resume=args.resume,
    )
    tracker.class_names = ds.class_names

    # ---- Resume ---------------------------------------------------------
    start_epoch = 0
    if args.resume:
        latest = tracker.find_latest_checkpoint()
        if latest and os.path.isfile(latest):
            keras_model.load_weights(latest)
            start_epoch = tracker.parse_epoch_from_checkpoint(latest) + 1
            log.info("Resumed from %s -> starting at epoch %d",
                     latest, start_epoch)
        else:
            log.info("No checkpoint found for resume — starting from scratch")

    # ---- Optimizer (AdamW — best practice) --------------------------------
    import keras
    optimizer = keras.optimizers.AdamW(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )

    # ---- LR schedules (per parameter group) ------------------------------
    steps_per_epoch = max(
        1, math.ceil(len(train_indices) / args.batch_size)
    )
    total_epochs = args.epochs

    lr_schedules = build_param_group_schedules(
        schedule_name=args.lr_scheduler,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        lr_component_decay=args.lr_component_decay,
        total_epochs=total_epochs,
        steps_per_epoch=steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        train_mode=args.train_mode,
        lr_drop=args.lr_drop,
        milestones=args.lr_milestones,
        gamma=args.lr_gamma,
        lr_min_factor=args.lr_min_factor,
    )

    log.info("LR schedules built: %s with %.1f warmup epochs",
             args.lr_scheduler, args.warmup_epochs)
    log.info("  Steps per epoch   : %d", steps_per_epoch)
    last_step = max(0, steps_per_epoch * total_epochs - 1)
    log.info("  Head LR range     : %.2e -> %.2e",
             lr_schedules["head"](0),
             lr_schedules["head"](last_step))
    log.info("  Backbone LR range : %.2e -> %.2e",
             lr_schedules["backbone"](0),
             lr_schedules["backbone"](last_step))
    log.info("  Decoder LR range  : %.2e -> %.2e",
             lr_schedules["decoder"](0),
             lr_schedules["decoder"](last_step))

    # ---- EMA -------------------------------------------------------------
    ema_m = None
    if args.use_ema:
        from paz.models.detection.dino_v2_object_detection.utils.utils import (
            ModelEma,
        )
        ema_m = ModelEma(
            keras_model, decay=args.ema_decay, tau=args.ema_tau,
        )
        log.info("EMA enabled (decay=%.4f, tau=%.1f)",
                 args.ema_decay, args.ema_tau)

    # ---- Early stopping --------------------------------------------------
    early_stopper = None
    if args.early_stopping:
        early_stopper = EarlyStopper(
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
            restore_best_weights=True,
            logger=log,
        )
        log.info("Early stopping enabled (patience=%d, min_delta=%.1e)",
                 args.early_stopping_patience, args.early_stopping_min_delta)

    # ---- Checkpointing state ---------------------------------------------
    best_val_loss = float("inf")
    prev_best_ckpt_path = None

    # =====================================================================
    # TRAINING LOOP
    # =====================================================================
    log.info("")
    log.info("=" * 68)
    log.info("STARTING TRAINING")
    log.info("  Epochs       : %d (start=%d)", total_epochs, start_epoch)
    log.info("  Batch size   : %d", args.batch_size)
    log.info("  Train mode   : %s", args.train_mode)
    log.info("  LR scheduler : %s", args.lr_scheduler)
    log.info("  Variant      : %s", args.variant)
    log.info("  Data loader  : DetectionDataGenerator "
             "(workers=%d, prefetch=%d)",
             args.num_workers, args.prefetch_size)
    log.info("=" * 68)
    log.info("")

    # ---- Build reusable training data generator -------------------------
    train_gen = DetectionDataGenerator(
        dataset=train_ds,
        indices=train_indices,
        batch_size=args.batch_size,
        augmentation=args.augmentation,
        seed=args.seed,
        shuffle=True,
        workers=args.num_workers,
        max_queue_size=args.prefetch_size,
    )

    global_step = start_epoch * steps_per_epoch
    training_start_time = time.time()

    for epoch in range(start_epoch, total_epochs):
        epoch_start_time = time.time()

        log.info("=" * 60)
        log.info("Epoch %d / %d", epoch, total_epochs - 1)
        log.info("=" * 60)

        # ---- Build training data iterator for this epoch ----------------
        train_gen.set_epoch(epoch)
        train_iter = prefetch_iterator(
            train_gen, max_prefetch=args.prefetch_size,
        )

        # ---- Train one epoch --------------------------------------------
        train_stats = train_one_epoch_custom(
            model=keras_model,
            criterion=criterion,
            optimizer=optimizer,
            data_iterator=train_iter,
            num_steps=steps_per_epoch,
            epoch=epoch,
            clip_max_norm=args.clip_max_norm,
            lr_schedules=lr_schedules,
            global_step=global_step,
            train_mode=args.train_mode,
            print_freq=args.print_freq,
            logger=log,
        )

        global_step = train_stats.get(
            "global_step", global_step + steps_per_epoch
        )

        # ---- EMA update -------------------------------------------------
        if ema_m is not None:
            ema_m.update(keras_model)

        # ---- Validation -------------------------------------------------
        val_metrics = {}
        train_eval_metrics = {}
        if args.validate and val_criterion is not None and val_ds is not None:
            # Evaluate on VALIDATION set
            log.info("  Running evaluation on VAL set...")
            val_t0 = time.time()
            val_metrics = validate_epoch_full(
                model=keras_model,
                criterion=val_criterion,
                dataset=val_ds,
                indices=val_indices,
                batch_size=args.batch_size,
                num_classes=ds.num_classes,
                class_names=ds.class_names,
                conf_threshold=args.confidence_threshold,
                iou_threshold=args.iou_threshold,
                max_batches=args.max_batches,
                logger=log,
                prefix="val",
            )
            val_elapsed = time.time() - val_t0
            log.info("  Val evaluation completed in %.1fs", val_elapsed)

            # Evaluate on TRAINING set (monitor overfitting)
            log.info("  Running evaluation on TRAIN set...")
            train_eval_t0 = time.time()
            train_eval_metrics = validate_epoch_full(
                model=keras_model,
                criterion=val_criterion,
                dataset=train_ds,
                indices=train_indices,
                batch_size=args.batch_size,
                num_classes=ds.num_classes,
                class_names=ds.class_names,
                conf_threshold=args.confidence_threshold,
                iou_threshold=args.iou_threshold,
                max_batches=args.max_batches,
                logger=None,
                prefix="train",
            )
            train_eval_elapsed = time.time() - train_eval_t0
            log.info("  Train evaluation completed in %.1fs",
                     train_eval_elapsed)

        # ---- Extract metrics --------------------------------------------
        train_loss = train_stats.get("train_loss", 0.0)
        val_loss = val_metrics.get("val_loss", 0.0)

        # Detection metrics — validation
        val_mAP_50 = val_metrics.get("val_mAP_50", 0.0)
        val_mAP_50_95 = val_metrics.get("val_mAP_50_95", 0.0)
        val_precision = val_metrics.get("val_precision", 0.0)
        val_recall = val_metrics.get("val_recall", 0.0)
        val_f1 = val_metrics.get("val_f1", 0.0)
        val_accuracy = val_metrics.get("val_accuracy", 0.0)
        val_num_gt = val_metrics.get("val_num_gt_boxes", 0)
        val_num_pred = val_metrics.get("val_num_pred_boxes", 0)

        # Detection metrics — training (evaluated in inference mode)
        train_mAP_50 = train_eval_metrics.get("train_mAP_50", 0.0)
        train_mAP_50_95 = train_eval_metrics.get("train_mAP_50_95", 0.0)
        train_precision = train_eval_metrics.get("train_precision", 0.0)
        train_recall = train_eval_metrics.get("train_recall", 0.0)
        train_f1 = train_eval_metrics.get("train_f1", 0.0)
        train_accuracy = train_eval_metrics.get("train_accuracy", 0.0)
        train_num_gt = train_eval_metrics.get("train_num_gt_boxes", 0)
        train_num_pred = train_eval_metrics.get("train_num_pred_boxes", 0)

        grad_norm = train_stats.get("grad_norm", 0.0)
        grad_norm_max = train_stats.get("grad_norm_max", 0.0)
        train_loss_ce = train_stats.get("loss_ce", 0.0)
        train_loss_bbox = train_stats.get("loss_bbox", 0.0)
        train_loss_giou = train_stats.get("loss_giou", 0.0)
        val_loss_ce = val_metrics.get("val_loss_ce", 0.0)
        val_loss_bbox = val_metrics.get("val_loss_bbox", 0.0)
        val_loss_giou = val_metrics.get("val_loss_giou", 0.0)
        lr_backbone = train_stats.get("lr_backbone", 0.0)
        lr_decoder = train_stats.get("lr_decoder", 0.0)
        lr_head = train_stats.get("lr_head", 0.0)

        # ---- Per-epoch summary log --------------------------------------
        log.info("")
        log.info("-" * 60)
        log.info("Epoch %d Summary", epoch)
        log.info("-" * 60)
        log.info("  LOSSES:")
        log.info("    Train Loss (total) : %.4f", train_loss)
        log.info("    Val Loss   (total) : %.4f", val_loss)
        log.info("    Train loss_ce      : %.4f", train_loss_ce)
        log.info("    Train loss_bbox    : %.4f", train_loss_bbox)
        log.info("    Train loss_giou    : %.4f", train_loss_giou)
        log.info("    Val   loss_ce      : %.4f", val_loss_ce)
        log.info("    Val   loss_bbox    : %.4f", val_loss_bbox)
        log.info("    Val   loss_giou    : %.4f", val_loss_giou)
        log.info("  OPTIMIZATION:")
        log.info("    Gradient norm (avg): %.4f", grad_norm)
        log.info("    Gradient norm (max): %.4f", grad_norm_max)
        log.info("    LR backbone        : %.2e", lr_backbone)
        log.info("    LR decoder         : %.2e", lr_decoder)
        log.info("    LR head            : %.2e", lr_head)
        log.info("  TRAIN EVALUATION:")
        log.info("    mAP@50             : %.4f", train_mAP_50)
        log.info("    mAP@50:95          : %.4f", train_mAP_50_95)
        log.info("    Precision          : %.4f", train_precision)
        log.info("    Recall             : %.4f", train_recall)
        log.info("    F1 Score           : %.4f", train_f1)
        log.info("    Accuracy           : %.4f", train_accuracy)
        log.info("    GT Boxes           : %d", train_num_gt)
        log.info("    Pred Boxes         : %d", train_num_pred)
        log.info("  VAL EVALUATION:")
        log.info("    mAP@50             : %.4f", val_mAP_50)
        log.info("    mAP@50:95          : %.4f", val_mAP_50_95)
        log.info("    Precision          : %.4f", val_precision)
        log.info("    Recall             : %.4f", val_recall)
        log.info("    F1 Score           : %.4f", val_f1)
        log.info("    Accuracy           : %.4f", val_accuracy)
        log.info("    GT Boxes           : %d", val_num_gt)
        log.info("    Pred Boxes         : %d", val_num_pred)
        log.info("-" * 60)

        # ---- Metrics tracking -------------------------------------------
        tracker.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_mAP_50=val_mAP_50,
            val_mAP_50_95=val_mAP_50_95,
            val_precision=val_precision,
            val_recall=val_recall,
            val_f1=val_f1,
            val_accuracy=val_accuracy,
            val_num_gt_boxes=val_num_gt,
            val_num_pred_boxes=val_num_pred,
            train_mAP_50=train_mAP_50,
            train_mAP_50_95=train_mAP_50_95,
            train_precision=train_precision,
            train_recall=train_recall,
            train_f1=train_f1,
            train_accuracy=train_accuracy,
            train_num_gt_boxes=train_num_gt,
            train_num_pred_boxes=train_num_pred,
            learning_rate=lr_head,
            per_class_precision=val_metrics.get("per_class_precision"),
            per_class_recall=val_metrics.get("per_class_recall"),
            per_class_f1=val_metrics.get("per_class_f1"),
            per_class_ap50=val_metrics.get("per_class_ap50"),
            # Extended metrics
            grad_norm=grad_norm,
            grad_norm_max=grad_norm_max,
            train_loss_ce=train_loss_ce,
            train_loss_bbox=train_loss_bbox,
            train_loss_giou=train_loss_giou,
            val_loss_ce=val_loss_ce,
            val_loss_bbox=val_loss_bbox,
            val_loss_giou=val_loss_giou,
            lr_backbone=lr_backbone,
            lr_decoder=lr_decoder,
            lr_head=lr_head,
        )

        # ---- Checkpointing ---------------------------------------------
        monitored = val_loss if args.validate else train_loss
        ckpt_path = tracker.checkpoint_path(epoch, val_loss, val_mAP_50)

        if args.checkpoint_mode == "every":
            keras_model.save_weights(ckpt_path)
            log.info("  [Checkpoint] Saved (every epoch): %s", ckpt_path)

        elif args.checkpoint_mode == "best_keep":
            if monitored < best_val_loss:
                best_val_loss = monitored
                keras_model.save_weights(ckpt_path)
                log.info("  [Checkpoint] NEW BEST (loss=%.4f): %s",
                         monitored, ckpt_path)
                best_path = tracker.best_checkpoint_path()
                keras_model.save_weights(best_path)
                log.info("  [Checkpoint] Updated best: %s", best_path)
            else:
                log.info("  [Checkpoint] No improvement "
                         "(current=%.4f, best=%.4f) — skipped",
                         monitored, best_val_loss)

        elif args.checkpoint_mode == "best_replace":
            if monitored < best_val_loss:
                if prev_best_ckpt_path and os.path.isfile(prev_best_ckpt_path):
                    os.remove(prev_best_ckpt_path)
                    log.info("  [Checkpoint] Deleted previous: %s",
                             prev_best_ckpt_path)
                best_val_loss = monitored
                keras_model.save_weights(ckpt_path)
                prev_best_ckpt_path = ckpt_path
                log.info("  [Checkpoint] NEW BEST (loss=%.4f): %s",
                         monitored, ckpt_path)
                best_path = tracker.best_checkpoint_path()
                keras_model.save_weights(best_path)
            else:
                log.info("  [Checkpoint] No improvement "
                         "(current=%.4f, best=%.4f) — skipped",
                         monitored, best_val_loss)

        # ---- Plots ------------------------------------------------------
        if tracker.should_plot(epoch, total_epochs):
            tracker.generate_plots()
            log.info("  [Plots] Updated: %s",
                     os.path.join(exp_dir, "plots"))

        # ---- Early stopping ---------------------------------------------
        if early_stopper is not None:
            should_stop = early_stopper.step(
                value=monitored,
                epoch=epoch,
                model=keras_model,
                mAP_50_95=val_mAP_50_95,
            )
            if should_stop:
                early_stopper.restore_weights(keras_model)
                break

        epoch_elapsed = time.time() - epoch_start_time
        log.info("  Epoch %d total time: %s",
                 epoch, str(datetime.timedelta(seconds=int(epoch_elapsed))))
        log.info("")

    # =====================================================================
    # POST-TRAINING
    # =====================================================================
    total_time = time.time() - training_start_time
    log.info("")
    log.info("=" * 68)
    log.info("TRAINING COMPLETE")
    log.info("=" * 68)
    log.info("  Total training time : %s",
             str(datetime.timedelta(seconds=int(total_time))))

    # Apply EMA weights if used
    if ema_m is not None:
        ema_m.apply_to(keras_model)
        log.info("  EMA weights applied to model")

    # Final plots
    tracker.generate_plots()
    log.info("  Final plots saved to %s", os.path.join(exp_dir, "plots"))

    # Summary
    log.info("")
    log.info("  Best monitored loss : %.4f", best_val_loss)
    if early_stopper is not None:
        log.info("  Early stopping best epoch  : %d",
                 early_stopper.best_epoch)
        log.info("  Early stopping best loss   : %.6f",
                 early_stopper.best_value)
        log.info("  Early stopping best mAP    : %.4f",
                 early_stopper.best_mAP)
        if early_stopper.stopped_epoch >= 0:
            log.info("  Early stopping triggered at: epoch %d",
                     early_stopper.stopped_epoch)

    log.info("  Experiment directory: %s", exp_dir)
    log.info("  Checkpoints        : %s",
             os.path.join(exp_dir, "checkpoints"))
    log.info("  Plots              : %s",
             os.path.join(exp_dir, "plots"))
    log.info("  Metrics log        : %s", tracker.log_path)

    if tracker.history["epoch"]:
        log.info("\n%s", tracker.format_epoch_summary(-1))
    log.info("=" * 68)


if __name__ == "__main__":
    main()
