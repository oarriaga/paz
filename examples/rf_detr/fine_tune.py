"""Fine-tunes a COCO RF-DETR on VOC2007.

The COCO weights load into every layer except the two class heads, whose
width depends on the class count, so they start fresh and the rest of the
detector keeps what it learned. Pass --freeze_backbone to train only the
projector, decoder and heads, or --lora_rank to adapt the backbone through
low rank adapters instead.

Every run writes a CSV log and a metrics figure. --tensorboard adds the
Keras TensorBoard callback, which needs tensorboard installed.

Training follows the reference recipe: every decoder layer and the first
stage are scored, the backbone trains slower the deeper the block, and
AdamW clips its gradients, optionally accumulates them over several batches,
keeps an exponential moving average of the weights and decays the rate on a
cosine after a warmup.

Fine-tuning uses a single query group. Upstream trains with thirteen and
drops all but the first at inference. --num_groups builds more, but the
published weights only carry the first, so the others start fresh and
roughly double the loss the run opens at.
"""
import argparse
import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras

import paz
from generator import Generator, preprocess_batch

BACKBONE_PREFIXES = ("patch_embed", "cls_token", "pos_embed", "block_",
                     "norm")
# DINOv2-small depth. Keras prunes the blocks past the last tapped one, so
# the built model does not carry all twelve.
NUM_BACKBONE_BLOCKS = 12
# Reference fine-tune rates, relative to the base learning rate.
ENCODER_RATE, LAYER_DECAY, COMPONENT_DECAY = 1.5, 0.8, 0.7
NO_DECAY = ("bias", "gamma", "beta", "pos_embed", "cls_token")
VARIANTS = {
    "nano": paz.models.TrainableRFDETRNano,
    "small": paz.models.TrainableRFDETRSmall,
    "medium": paz.models.TrainableRFDETRMedium,
    "base": paz.models.TrainableRFDETRBase,
    "large": paz.models.TrainableRFDETRLarge,
}


def build_model(variant, num_classes, num_groups):
    """Loads COCO weights into every layer whose shape still matches.

    Keras keys a weights file by layer class and position rather than by
    name, so the layers an extra query group adds would shift everything
    after them and misplace most of the file. The weights therefore load
    into a single-group detector and are copied by name from there, which
    leaves the added groups on their fresh initialization.
    """
    source = VARIANTS[variant](num_classes, 1)
    weights = paz.models.detection.rf_detr.download_weights(source)
    source.load_weights(weights, skip_mismatch=True)
    paz.models.detection.rf_detr.reset_class_heads(source)
    if num_groups == 1:
        model = source
    else:
        model = VARIANTS[variant](num_classes, num_groups)
        copy_weights(source, model)
        # The added groups own first-stage heads the copy never reaches.
        paz.models.detection.rf_detr.reset_class_heads(model)
    return model


def copy_weights(source, model):
    """Copies every weighted layer the two models name alike."""
    for layer in source.layers:
        if layer.weights:
            model.get_layer(layer.name).set_weights(layer.get_weights())


def freeze_backbone(model):
    """Trains the projector, decoder and heads only."""
    for layer in model.layers:
        if layer.name.startswith(BACKBONE_PREFIXES):
            layer.trainable = False


def adapt_backbone(model, rank):
    """Trains the backbone through LoRA adapters on its dense layers."""
    for layer in model.layers:
        if layer.name.startswith(BACKBONE_PREFIXES):
            adapt_layer(layer, rank)


def adapt_layer(layer, rank):
    """Dense layers gain adapters; everything else freezes."""
    if hasattr(layer, "enable_lora"):
        layer.enable_lora(rank)
    else:
        layer.trainable = False


def build_learning_rate_scales(model):
    """Rate factor per variable, as the reference recipe sets them."""
    scales = {}
    for variable in model.trainable_variables:
        scales[variable.path] = compute_scale(variable.path)
    return scales


def compute_scale(path):
    """Earlier blocks train slower, and the decoder slower than the heads."""
    if path.startswith(BACKBONE_PREFIXES):
        depth = NUM_BACKBONE_BLOCKS + 1 - read_block_index(path)
        scale = ENCODER_RATE * LAYER_DECAY**depth * COMPONENT_DECAY**2
    elif path.startswith("decoder"):
        scale = COMPONENT_DECAY
    else:
        scale = 1.0
    return scale


def read_block_index(path):
    """Zero for the patch embedding and ``i + 1`` for block ``i``."""
    if path.startswith("block_"):
        index = int(path.split("_")[1]) + 1
    elif path.startswith(("patch_embed", "cls_token", "pos_embed")):
        index = 0
    else:
        index = NUM_BACKBONE_BLOCKS + 1
    return index


def build_schedule(learning_rate, warmup_steps, decay_steps):
    """Linear warmup, then a cosine decay to a tenth of the rate."""
    kwargs = dict(warmup_target=learning_rate, warmup_steps=warmup_steps)
    args = learning_rate / 100.0, decay_steps
    return keras.optimizers.schedules.CosineDecay(*args, alpha=0.1, **kwargs)


def build_optimizer(model, schedule, weight_decay, clipnorm, accumulation,
                    ema_momentum):
    """AdamW with per-variable rates, clipping, accumulation and an EMA."""
    steps = accumulation if accumulation > 1 else None
    kwargs = dict(weight_decay=weight_decay, global_clipnorm=clipnorm,
                  gradient_accumulation_steps=steps, use_ema=True,
                  ema_momentum=ema_momentum)
    scales = build_learning_rate_scales(model)
    optimizer = paz.optimizers.LayerwiseAdamW(scales, schedule, **kwargs)
    optimizer.exclude_from_weight_decay(var_names=NO_DECAY)
    return optimizer


def build_generators(key, resolution, batch_size, max_boxes, workers):
    train_data = paz.datasets.voc.load("VOC2007", "trainval")
    test_data = paz.datasets.voc.load("VOC2007", "test")
    args = resolution, max_boxes
    train_pipeline = paz.lock(preprocess_batch, *args, True)
    test_pipeline = paz.lock(preprocess_batch, *args, False)
    train_args = key, *train_data, batch_size, train_pipeline, workers
    test_args = key, *test_data, batch_size, test_pipeline, workers
    return Generator(*train_args), Generator(*test_args)


def build_evaluation(model, num_classes, score_thresh, period, num_images):
    """Scores VOC mAP on a slice of the test split every few epochs."""
    detector = paz.models.detection.rf_detr.build_detector(model)
    detect = paz.applications.detectors.RFDETR(detector, score_thresh, None)
    images, ground_truths = paz.datasets.voc.load("VOC2007", "test")
    difficulties = paz.datasets.voc.load_difficulties("VOC2007", "test")
    args = detect, images[:num_images], ground_truths[:num_images]
    kwargs = dict(difficulties=difficulties[:num_images])
    return paz.callbacks.EvaluateMAP(*args, num_classes, period, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="nano", choices=list(VARIANTS))
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--clipnorm", default=0.1, type=float)
    parser.add_argument("--accumulation_steps", default=1, type=int)
    parser.add_argument("--ema_momentum", default=0.993, type=float)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--warmup_epochs", default=1, type=int)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--max_num_boxes", default=25, type=int)
    parser.add_argument("--num_groups", default=1, type=int)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--lora_rank", default=0, type=int)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--evaluation_period", default=0, type=int)
    parser.add_argument("--evaluation_images", default=500, type=int)
    parser.add_argument("--score_thresh", default=0.05, type=float)
    parser.add_argument("--steps_per_epoch", default=None, type=int)
    parser.add_argument("--validation_steps", default=None, type=int)
    parser.add_argument("--root", default="experiments")
    parser.add_argument("--label", default=None)
    args = parser.parse_args()
    root, key = paz.logger.setup(args)

    names = paz.datasets.voc.get_class_names()
    model = build_model(args.variant, len(names), args.num_groups)
    if args.freeze_backbone:
        freeze_backbone(model)
    if args.lora_rank > 0:
        adapt_backbone(model, args.lora_rank)
    resolution = model.input_shape[1]

    generator_args = (key, resolution, args.batch_size, args.max_num_boxes, 1)
    train_generator, test_generator = build_generators(*generator_args)
    steps_per_epoch = args.steps_per_epoch or len(train_generator)
    warmup_steps = steps_per_epoch * args.warmup_epochs
    decay_steps = max(steps_per_epoch * args.epochs - warmup_steps, 1)
    schedule = build_schedule(args.learning_rate, warmup_steps, decay_steps)
    optimizer_args = (schedule, args.weight_decay, args.clipnorm,
                      args.accumulation_steps, args.ema_momentum)
    optimizer = build_optimizer(model, *optimizer_args)

    metrics = [
        paz.losses.detr.classification,
        paz.losses.detr.regression,
        paz.losses.detr.generalized_IOU,
    ]
    checkpoint = os.path.join(root, f"rf_detr_{args.variant}_voc.weights.h5")
    kwargs = dict(save_weights_only=True, save_best_only=True, verbose=1)
    save = keras.callbacks.ModelCheckpoint(checkpoint, **kwargs)
    log = keras.callbacks.CSVLogger(os.path.join(root, "optimization.log"))
    kwargs = dict(min_delta=1e-3, restore_best_weights=True, verbose=1)
    stop = keras.callbacks.EarlyStopping(patience=args.patience, **kwargs)
    plot = paz.callbacks.PlotMetrics(os.path.join(root, "metrics.png"))
    callbacks = [save]
    if args.evaluation_period > 0:
        evaluation_args = (model, len(names), args.score_thresh,
                           args.evaluation_period, args.evaluation_images)
        # Ahead of the log and the figure, which read the epoch it fills in.
        callbacks.append(build_evaluation(*evaluation_args))
    callbacks = callbacks + [log, stop, plot]
    if args.tensorboard:
        board = os.path.join(root, "tensorboard")
        callbacks.append(keras.callbacks.TensorBoard(board))

    # The Hungarian assignment is a jax.pure_callback, so it traces inside the
    # compiled train step; eager execution of this graph is far slower.
    losses = paz.losses.detr.call
    model.compile(optimizer, losses, metrics=metrics, jit_compile=True)
    kwargs = dict(epochs=args.epochs, callbacks=callbacks,
                  steps_per_epoch=args.steps_per_epoch,
                  validation_data=test_generator,
                  validation_steps=args.validation_steps)
    model.fit(train_generator, **kwargs)
    # Keras averages the weights once training ends, but early stopping then
    # restores the best epoch's raw ones, so average them again here.
    optimizer.finalize_variable_values(model.trainable_variables)
    model.save_weights(checkpoint.replace(".weights", "_ema.weights"))
