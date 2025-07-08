import os
import argparse

os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"

import paz
import keras
from generator import Generator
from pipeline2 import preprocess_batch

parser = argparse.ArgumentParser(description="Training script for SSD on VOC")
parser.add_argument("--seed", default=777, type=int)
parser.add_argument("--model", default="SSD512", type=str)
parser.add_argument("--root", default="experiments", type=str)
parser.add_argument("--label", default=None)
parser.add_argument("--trainable_base", default=False, type=bool)
parser.add_argument("--model_path", default=None)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--num_workers", default="max")
parser.add_argument("--max_queue_size", default=75, type=float)
parser.add_argument("--decay_epochs", nargs="+", type=int, default=[110, 152])
parser.add_argument("--decay_rate", default=0.1, type=float)
parser.add_argument("--max_num_epochs", default=240, type=int)
parser.add_argument("--max_num_boxes", default=25, type=int)
parser.add_argument("--match_IOU", default=0.5, type=float)
parser.add_argument("--box_variances", nargs="+", default=[0.1, 0.1, 0.2, 0.2])
args = parser.parse_args()
root, key = paz.logger.setup(args)

mean = paz.image.BGR_IMAGENET_MEAN
train_data = paz.datasets.deepfish.load("train")
test_data = paz.datasets.deepfish.load("validation")
num_classes = len(paz.datasets.deepfish.get_class_names())
model_to_class = {"SSD300": paz.models.SSD300, "SSD512": paz.models.SSD512}
if args.model == "SSD300":
    H, W, base = 300, 300, "VOC"
    model_args = [num_classes + 1, base, None, (H, W, 3)]
elif args.model == "SSD512":
    H, W, base = 512, 512, "COCO"
    model_args = [num_classes + 1, base, None, (H, W, 3)]
else:
    raise ValueError(f"Model {args.model} is not supported.")

prior_boxes = paz.models.detection.single_shot_detector.build_prior_boxes(base)
Model = model_to_class[args.model]
model = Model(*model_args, trainable_base=args.trainable_base)

if args.model_path is not None:
    model.load_weights(filepath=args.model_path)

model.summary()

metrics = {
    "boxes": [
        paz.losses.multibox.regression,
        paz.losses.multibox.positive_classification,
        paz.losses.multibox.negative_classification,
    ]
}

checkpoint = os.path.join(root, f"{args.model}.keras")
callbacks = [
    keras.callbacks.ModelCheckpoint(checkpoint, verbose=1, save_best_only=True),
    keras.callbacks.CSVLogger(os.path.join(root, "optimization.log")),
    paz.callbacks.EpochScheduler(args.decay_epochs, args.decay_rate),
]

optimizer = keras.optimizers.SGD(args.learning_rate, args.momentum)
model.compile(
    optimizer, paz.losses.multibox.call, metrics=metrics, jit_compile=True
)
batch_args = (
    H,
    W,
    prior_boxes,
    num_classes,
    args.match_IOU,
    args.box_variances,
    mean,
    args.max_num_boxes,
)

num_workers = os.cpu_count() if args.num_workers == "max" else args.num_workers
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
    epochs=args.max_num_epochs,
    validation_data=valid_generator,
    callbacks=callbacks,
)
