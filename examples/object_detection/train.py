import os
import math
import argparse

# SSD300 with the paper recipe is only marginally stable: fused/compiled kernel
# execution (JAX/XLA or torch.compile) reorders float reductions just enough to
# tip it into a NaN runaway, while eager execution stays on the stable side. So
# train eager on the torch backend. JAX runs the augmentation on CPU so it does
# not preallocate the GPU that the torch model needs.
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import paz
import keras
from generator import Generator, preprocess_batch

parser = argparse.ArgumentParser(description="Training script for SSD on VOC")
parser.add_argument("--seed", default=777, type=int)
parser.add_argument("--model", default="SSD300", type=str)
parser.add_argument("--root", default="experiments", type=str)
parser.add_argument("--label", default=None)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--l2_loss", default=0.0, type=float)
parser.add_argument("--num_workers", default="max")
parser.add_argument("--max_queue_size", default=50, type=int)
parser.add_argument("--decay_iterations", nargs="+", type=int,
                    default=[80000, 100000])
parser.add_argument("--decay_rate", default=0.1, type=float)
parser.add_argument("--num_iterations", default=120000, type=int)
parser.add_argument("--H", default=300, type=int, help="Height of input images")
parser.add_argument("--W", default=300, type=int, help="Width of input images")
parser.add_argument("--max_num_boxes", default=25, type=int)
parser.add_argument("--match_IOU", default=0.5, type=float)
parser.add_argument("--box_variances", nargs="+", default=[0.1, 0.1, 0.2, 0.2])
parser.add_argument("--eval_period", default=10, type=int)
args = parser.parse_args()
root, key = paz.logger.setup(args)

prior_boxes = paz.models.detection.single_shot_detector.build_prior_boxes("VOC")
num_classes = len(paz.datasets.labels("VOC"))
mean = paz.image.BGR_IMAGENET_MEAN
images_07, detections_07 = paz.datasets.load("VOC2007", "trainval")
images_12, detections_12 = paz.datasets.load("VOC2012", "trainval")
train_data = (images_07 + images_12, detections_07 + detections_12)
# The SSD paper / amdegroot schedule is defined in iterations (decay at 80k and
# 100k, stop at 120k). Convert to epochs for the epoch scheduler so the recipe
# stays aligned to the baseline regardless of batch size.
steps_per_epoch = math.ceil(len(train_data[0]) / args.batch_size)
decay_epochs = [round(i / steps_per_epoch) for i in args.decay_iterations]
num_epochs = round(args.num_iterations / steps_per_epoch)
test_data = paz.datasets.load("VOC2007", "test")
input_shape = (args.H, args.W, 3)
model = paz.models.SSD300(
    num_classes + 1, "VGG", None, input_shape, l2_loss=args.l2_loss,
    trainable_base=True
)
model.summary()

nms = paz.lock(paz.detection.apply_per_class_NMS, num_classes, 0.45, 200)
detector_args = model, 0.01, prior_boxes, args.box_variances, nms, None
detector = paz.applications.detectors.SSD(*detector_args)
test_images, test_detections = test_data
difficulties = paz.datasets.voc.load_difficulties("VOC2007", "test")

metrics = {
    "boxes": [
        paz.losses.multibox.regression,
        paz.losses.multibox.positive_classification,
        paz.losses.multibox.negative_classification,
    ]
}

map_args = (detector, test_images, test_detections, num_classes,
            args.eval_period, difficulties)
checkpoint = os.path.join(root, f"{args.model}.keras")
callbacks = [
    keras.callbacks.ModelCheckpoint(checkpoint, verbose=1, save_best_only=True),
    keras.callbacks.CSVLogger(os.path.join(root, "optimization.log")),
    paz.callbacks.EpochScheduler(decay_epochs, args.decay_rate),
    paz.callbacks.EvaluateMAP(*map_args),
]

optimizer = keras.optimizers.SGD(args.learning_rate, args.momentum,
                                 weight_decay=args.weight_decay)
# jit_compile=False keeps execution eager (see the header note on stability).
model.compile(
    optimizer, paz.losses.multibox.call, metrics=metrics, jit_compile=False
)
batch_args = (
    args.H,
    args.W,
    prior_boxes,
    num_classes,
    args.match_IOU,
    args.box_variances,
    mean,
    args.max_num_boxes,
)

use_all_cpus = args.num_workers == "max"
num_workers = os.cpu_count() if use_all_cpus else int(args.num_workers)
train_pipeline = paz.lock(preprocess_batch, *batch_args, True)
train_generator = Generator(
    key,
    *train_data,
    args.batch_size,
    train_pipeline,
    num_workers,
    args.max_queue_size,
)

valid_pipeline = paz.lock(preprocess_batch, *batch_args, False)

valid_generator = Generator(
    key,
    *test_data,
    args.batch_size,
    valid_pipeline,
    num_workers,
    args.max_queue_size,
)

history = model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=valid_generator,
    callbacks=callbacks,
)
