import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import argparse

import numpy as np
import keras
from keras.optimizers import Adam

import paz
from paz.losses import multibox
from paz.losses import MultiPoseLoss
from paz.models import EfficientPosePhi0
from linemod import Linemod, LINEMOD_CAMERA_MATRIX, RGB_LINEMOD_MEAN
from linemod import load_model_points


def compile_model(model, model_points, translation_priors):
    pose_loss = MultiPoseLoss(model_points, translation_priors,
                              LINEMOD_CAMERA_MATRIX)
    losses = {"boxes": multibox.call, "transformation": pose_loss.compute_loss}
    weights = {"boxes": 1.0, "transformation": 0.02}
    optimizer = Adam(learning_rate=1e-4, clipnorm=0.001)
    model.compile(optimizer=optimizer, loss=losses, loss_weights=weights)
    return model


def build_targets(sample, prior_boxes, translation_priors, num_classes,
                  input_size, scale=1.0):
    boxes = sample["boxes"]
    box_target = build_box_targets(boxes, prior_boxes, num_classes)
    pose_target = build_pose_targets(boxes, sample["rotation"],
                                     sample["translation_raw"], prior_boxes,
                                     scale)
    return box_target, pose_target


def build_box_targets(boxes, prior_boxes, num_classes):
    matched = paz.detection.match(boxes, prior_boxes)
    encoded = paz.detection.encode(matched, prior_boxes)
    return np.asarray(paz.detection.to_one_hot(encoded, num_classes))


def build_pose_targets(boxes, rotations, translations, prior_boxes, scale):
    rotation = paz.poses.rotation_matrix_to_axis_angle(rotations)[:, :3]
    class_args = boxes[:, -1:]
    unused = np.zeros((len(boxes), 1))
    symmetric = np.zeros((len(boxes), 1))
    poses = np.concatenate(
        [rotation, symmetric, class_args, unused, translations], axis=1)
    matched = paz.poses.match_poses(boxes, poses, prior_boxes)
    return paz.poses.concatenate_scale(matched, np.float32(scale))


class SyntheticPoseData(keras.utils.PyDataset):
    """Random batches with a few positive anchors, to smoke-test the training
    wiring (model + losses + fit) without the LINEMOD dataset."""

    def __init__(self, num_boxes, num_classes, input_size, batch_size=2,
                 batches=2, num_positives=5):
        super().__init__()
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.input_size = input_size
        self.batch_size = batch_size
        self.batches = batches
        self.num_positives = num_positives

    def __len__(self):
        return self.batches

    def __getitem__(self, index):
        rng = np.random.default_rng(index)
        shape = (self.batch_size, self.input_size, self.input_size, 3)
        images = rng.normal(size=shape).astype("float32")
        box_target = self.make_box_target(rng)
        pose_target = self.make_pose_target(rng)
        return images, {"boxes": box_target, "transformation": pose_target}

    def make_box_target(self, rng):
        target = np.zeros((self.batch_size, self.num_boxes, 4 + self.num_classes))  # fmt: skip
        target[:, :, 4] = 1.0
        target[:, : self.num_positives, 4] = 0.0
        target[:, : self.num_positives, 5] = 1.0
        target[:, : self.num_positives, :4] = rng.normal(size=(self.batch_size, self.num_positives, 4))  # fmt: skip
        return target.astype("float32")

    def make_pose_target(self, rng):
        target = np.zeros((self.batch_size, self.num_boxes, 11))
        positives = self.num_positives
        target[:, :positives, 0:3] = rng.normal(size=(self.batch_size, positives, 3)) * 0.2  # fmt: skip
        target[:, :positives, 6:9] = rng.normal(size=(self.batch_size, positives, 3))  # fmt: skip
        target[:, :positives, -2] = 1.0
        target[:, :, -1] = 1.0
        return target.astype("float32")


def smoke_train():
    model = EfficientPosePhi0(num_classes=2, base_weights=None,
                              head_weights=None)
    model_points = np.random.default_rng(0).normal(size=(64, 3))
    compile_model(model, model_points, np.asarray(model.translation_priors))
    num_boxes = np.asarray(model.prior_boxes).shape[0]
    data = SyntheticPoseData(num_boxes, 2, model.input_shape[1])
    history = model.fit(data, epochs=1, verbose=0)
    print("smoke train loss:", float(history.history["loss"][0]))


def train_linemod(args):
    data_manager = Linemod(args.data_path, args.object_id, "train")
    dataset = data_manager.load_data()
    model = EfficientPosePhi0(num_classes=data_manager.num_classes,
                              base_weights="COCO", head_weights=None)
    model_points = load_model_points(args.data_path, args.object_id)
    compile_model(model, model_points, np.asarray(model.translation_priors))
    sequence = LinemodSequence(dataset, model, args.batch_size)
    model.fit(sequence, epochs=args.num_epochs)


class LinemodSequence(keras.utils.PyDataset):
    def __init__(self, dataset, model, batch_size):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.prior_boxes = np.asarray(model.prior_boxes)
        self.translation_priors = np.asarray(model.translation_priors)
        self.num_classes = model.output_shape[0][-1] - 4
        self.input_size = model.input_shape[1]
        self.mean = np.array(RGB_LINEMOD_MEAN)

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __getitem__(self, index):
        samples = self.dataset[index * self.batch_size:
                               (index + 1) * self.batch_size]
        images, box_targets, pose_targets = [], [], []
        for sample in samples:
            image, scale = self.load_image(sample["image"])
            box_target, pose_target = build_targets(
                sample, self.prior_boxes, self.translation_priors,
                self.num_classes, self.input_size, scale)
            images.append(image)
            box_targets.append(box_target)
            pose_targets.append(pose_target)
        targets = {"boxes": np.array(box_targets),
                   "transformation": np.array(pose_targets)}
        return np.array(images), targets

    def load_image(self, path):
        image = paz.image.load(path)
        resized = paz.image.resize_opencv(image, (self.input_size, self.input_size))  # fmt: skip
        scale = self.input_size / max(paz.image.get_size(image))
        return np.asarray(resized, "float32") - self.mean, scale


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data_path", default="Linemod_preprocessed/")
    parser.add_argument("--object_id", default="08")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_epochs", default=500, type=int)
    args = parser.parse_args()
    smoke_train() if args.smoke else train_linemod(args)
