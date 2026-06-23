import os
import sys

os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"

import argparse

import keras
from keras.optimizers import Adam

import paz
from openimages import OpenImagesV6Hand

# Reuse the proven SSD training pipeline from the object detection example.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "object_detection"))  # fmt: skip
from generator import Generator
from pipeline2 import preprocess_batch


def parse_arguments():
    parser = argparse.ArgumentParser(description="SSD512 hand detection train")
    parser.add_argument("--data_path", default="open-images-v6/")
    parser.add_argument("--root", default="experiments", type=str)
    parser.add_argument("--label", default=None)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--num_epochs", default=240, type=int)
    parser.add_argument("--input_size", default=512, type=int)
    parser.add_argument("--match_IOU", default=0.5, type=float)
    parser.add_argument("--max_num_boxes", default=25, type=int)
    parser.add_argument("--num_workers", default="max")
    parser.add_argument("--max_queue_size", default=50, type=int)
    parser.add_argument("--box_variances", nargs="+", default=[0.1, 0.1, 0.2, 0.2])  # fmt: skip
    return parser.parse_args()


def main():
    args = parse_arguments()
    root, key = paz.logger.setup(args)
    prior_boxes = paz.models.detection.single_shot_detector.build_prior_boxes("COCO")  # fmt: skip
    num_classes = 1  # foreground hand class; SSD512 adds the background class
    shape = (args.input_size, args.input_size, 3)
    model = paz.models.SSD512(num_classes + 1, None, None, shape)
    metrics = {"boxes": [paz.losses.multibox.regression,
                         paz.losses.multibox.positive_classification,
                         paz.losses.multibox.negative_classification]}
    model.compile(Adam(args.learning_rate), paz.losses.multibox.call,
                  metrics=metrics, jit_compile=True)

    workers = os.cpu_count() if args.num_workers == "max" else args.num_workers
    batch_args = (args.input_size, args.input_size, prior_boxes, num_classes,
                  args.match_IOU, args.box_variances,
                  paz.image.BGR_IMAGENET_MEAN, args.max_num_boxes)
    train = build_generator(args, key, "train", batch_args, workers)
    valid = build_generator(args, key, "validation", batch_args, workers)
    callbacks = [keras.callbacks.CSVLogger(os.path.join(root, "log.csv")),
                 keras.callbacks.ModelCheckpoint(
                     os.path.join(root, "SSD512Hand.keras"),
                     save_best_only=True)]
    model.fit(train, epochs=args.num_epochs, validation_data=valid,
              callbacks=callbacks)


def build_generator(args, key, split, batch_args, workers, augment=True):
    images, detections = OpenImagesV6Hand(args.data_path, split).load_data()
    pipeline = paz.lock(preprocess_batch, *batch_args, split == "train")
    return Generator(key, images, detections, args.batch_size, pipeline,
                     workers, args.max_queue_size)


if __name__ == "__main__":
    main()
