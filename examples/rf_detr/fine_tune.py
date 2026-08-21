"""Fine-tunes a COCO RF-DETR on VOC2007.

The COCO weights load into every layer except the two class heads, whose width
depends on the class count, so they start fresh and the rest of the detector
keeps what it learned. Pass --freeze_backbone to train only the projector,
decoder and heads, which is cheaper and usually enough on a small dataset.

Fine-tuning uses a single query group. Upstream trains with thirteen and drops
all but the first at inference; the extra groups speed up convergence but are
not needed to fine-tune, and this detector only builds the first one.
"""
import argparse
import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras

import paz
from generator import Generator, preprocess_batch

BACKBONE_PREFIXES = ("patch_embed", "cls_token", "pos_embed", "block_",
                     "norm")
VARIANTS = {
    "nano": paz.models.RFDETRNano,
    "small": paz.models.RFDETRSmall,
    "medium": paz.models.RFDETRMedium,
    "base": paz.models.RFDETRBase,
    "large": paz.models.RFDETRLarge,
}


def build_model(variant, num_classes):
    """Loads COCO weights into every layer whose shape still matches."""
    model = VARIANTS[variant](num_classes=num_classes)
    weights = paz.models.detection.rf_detr.download_weights(model)
    model.load_weights(weights, skip_mismatch=True)
    paz.models.detection.rf_detr.reset_class_heads(model)
    return paz.models.detection.rf_detr.join_outputs(model)


def build_generators(key, resolution, batch_size, max_boxes, workers):
    train_data = paz.datasets.voc.load("VOC2007", "trainval")
    test_data = paz.datasets.voc.load("VOC2007", "test")
    args = resolution, max_boxes
    train_pipeline = paz.lock(preprocess_batch, *args, True)
    test_pipeline = paz.lock(preprocess_batch, *args, False)
    train_args = key, *train_data, batch_size, train_pipeline, workers
    test_args = key, *test_data, batch_size, test_pipeline, workers
    return Generator(*train_args), Generator(*test_args)


def freeze_backbone(model):
    """Trains the projector, decoder and heads only."""
    for layer in model.layers:
        if layer.name.startswith(BACKBONE_PREFIXES):
            layer.trainable = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="nano", choices=list(VARIANTS))
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--max_num_boxes", default=25, type=int)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--steps_per_epoch", default=None, type=int)
    parser.add_argument("--validation_steps", default=None, type=int)
    parser.add_argument("--root", default="experiments")
    parser.add_argument("--label", default=None)
    args = parser.parse_args()
    root, key = paz.logger.setup(args)

    names = paz.datasets.voc.get_class_names()
    model = build_model(args.variant, len(names))
    if args.freeze_backbone:
        freeze_backbone(model)
    resolution = model.input_shape[1]

    metrics = [
        paz.losses.detr.classification,
        paz.losses.detr.regression,
        paz.losses.detr.generalized_IOU,
    ]
    checkpoint = os.path.join(root, f"rf_detr_{args.variant}_voc.weights.h5")
    callbacks = [
        keras.callbacks.ModelCheckpoint(checkpoint, verbose=1,
                                        save_weights_only=True),
        keras.callbacks.CSVLogger(os.path.join(root, "optimization.log")),
    ]

    optimizer = keras.optimizers.AdamW(args.learning_rate, weight_decay=1e-4)
    # The Hungarian assignment is a jax.pure_callback, so it traces inside the
    # compiled train step; eager execution of this graph is far slower.
    model.compile(optimizer, paz.losses.detr.call, metrics=metrics,
                  jit_compile=True)

    generator_args = (key, resolution, args.batch_size, args.max_num_boxes, 1)
    train_generator, test_generator = build_generators(*generator_args)
    history = model.fit(
        train_generator,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_data=test_generator,
        validation_steps=args.validation_steps,
        callbacks=callbacks,
    )
