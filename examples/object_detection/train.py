import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"
import argparse
import jax
import jax.numpy as jp
import paz
import keras
from generator import Generator
from pipeline import preprocess_batch

jax.config.update("jax_debug_nans", True)

parser = argparse.ArgumentParser(description="Training script for SSD on VOC")
parser.add_argument("--seed", default=777, type=int)
parser.add_argument("--model", default="SSD300", type=str)
parser.add_argument("--root", default="experiments", type=str)
parser.add_argument("--label", default=None)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--clipnorm", default=10.0, type=float)
parser.add_argument("--num_workers", default=None, type=int)
parser.add_argument("--max_queue_size", default=100, type=float)
parser.add_argument("--decay_epochs", nargs="+", type=int, default=[110, 152])
parser.add_argument("--decay_rate", default=0.1, type=float)
parser.add_argument("--max_num_epochs", default=240, type=int)
parser.add_argument("--H", default=300, type=int, help="Height of input images")
parser.add_argument("--W", default=300, type=int, help="Width of input images")
parser.add_argument("--max_num_boxes", default=25, type=int)
parser.add_argument("--match_IOU", default=0.5, type=float)
parser.add_argument("--box_variances", nargs="+", default=[0.1, 0.1, 0.2, 0.2])
args = parser.parse_args()

root, key = paz.logger.setup(args)

prior_boxes = paz.models.detection.utils.create_prior_boxes("VOC")
num_classes = len(paz.datasets.labels("VOC"))
mean = jp.array(paz.image.BGR_IMAGENET_MEAN)
images_07, boxes_07, class_args_07 = paz.datasets.load("VOC2007", "trainval")
images_12, boxes_12, class_args_12 = paz.datasets.load("VOC2012", "trainval")
train_images = images_07 + images_12
train_boxes = boxes_07 + boxes_12
train_class_args = class_args_07 + class_args_12
train_data = (train_images, train_boxes, train_class_args)
test_data = paz.datasets.load("VOC2007", "test")

model = paz.models.SSD300(
    num_classes + 1, base_weights="VGG", head_weights=None, trainable_base=False
)
model.summary()

metrics = [
    paz.losses.multibox.regression,
    paz.losses.multibox.positive_classification,
    paz.losses.multibox.negative_classification,
]

checkpoint = os.path.join(root, args.model + ".keras")
callbacks = [
    keras.callbacks.ModelCheckpoint(checkpoint, verbose=1, save_best_only=True),
    keras.callbacks.CSVLogger(os.path.join(root, "optimization.log")),
    paz.callbacks.EpochScheduler(args.decay_epochs, args.decay_rate),
]

optimizer = keras.optimizers.SGD(
    args.learning_rate, args.momentum, global_clipnorm=args.clipnorm
)
model.compile(
    optimizer,
    loss=paz.losses.multibox.call,
    metrics=metrics,
    run_eagerly=False,
    jit_compile=True,
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

num_workers = os.cpu_count() if args.num_workers is None else args.num_workers
train_pipeline = paz.lock(preprocess_batch, *batch_args, True)
train_generator = Generator(
    key,
    *train_data,
    args.batch_size,
    train_pipeline,
    num_workers,
    args.max_queue_size
)
valid_pipeline = paz.lock(preprocess_batch, *batch_args, False)
valid_generator = Generator(
    key,
    *test_data,
    args.batch_size,
    valid_pipeline,
    num_workers,
    args.max_queue_size
)

model.fit(
    train_generator, epochs=args.max_num_epochs, validation_data=valid_generator
)
