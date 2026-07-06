import os
from collections import namedtuple
from functools import partial

import numpy as np
import cv2
import jax
import jax.numpy as jp
import keras

import paz
from paz.applications.pose_estimators import solve_PnP_RANSAC

Camera = namedtuple("Camera", ["intrinsics", "distortion"])


def render_views(render_image, render_coordinates, poses):
    images, coordinates, masks = [], [], []
    for pose in poses:
        nocs, mask = render_coordinates(pose)
        images.append(np.asarray(render_image(pose)))
        coordinates.append(np.asarray(nocs))
        masks.append(np.asarray(mask))
    images = np.stack(images).astype("uint8")
    coordinates = np.stack(coordinates).astype("float32")
    masks = np.stack(masks).astype("float32")
    return images, coordinates, masks


def load_backgrounds(path, image_size, num_backgrounds=None):
    if path is None:
        return None
    files = sorted(os.path.join(path, name) for name in os.listdir(path))
    files = files if num_backgrounds is None else files[:num_backgrounds]
    size = (image_size, image_size)
    crops = [paz.image.resize_opencv(paz.image.load(name), size)
             for name in files]
    return np.stack(crops).astype("uint8")


class Pix2PoseSequence(keras.utils.PyDataset):
    def __init__(self, images, coordinates, masks, batch_size,
                 backgrounds=None, num_occlusions=1, seed=0):
        super().__init__()
        self.images = images
        self.coordinates = coordinates
        self.masks = masks
        self.batch_size = batch_size
        self.key = jax.random.PRNGKey(seed)
        backgrounds = None if backgrounds is None else jp.asarray(backgrounds)
        randomize = partial(paz.image.randomize_rendered_image,
                            backgrounds=backgrounds,
                            num_occlusions=num_occlusions)
        self.randomize = jax.jit(jax.vmap(randomize, in_axes=(0, 0, 0)))

    def __len__(self):
        return len(self.images) // self.batch_size

    def __getitem__(self, index):
        start = index * self.batch_size
        stop = start + self.batch_size
        self.key, batch_key = jax.random.split(self.key)
        keys = jax.random.split(batch_key, self.batch_size)
        masks = self.masks[start:stop].astype("float32")
        inputs = self.randomize(keys, self.images[start:stop], masks)
        coordinates = self.coordinates[start:stop].astype("float32") / 255.0
        labels = np.concatenate([coordinates, masks[..., None]], axis=-1)
        return np.asarray(inputs) / 255.0, labels


def build_camera(y_FOV, size):
    H, W = size
    focal = (1.0 / np.tan(y_FOV / 2.0)) * (H / 2.0)
    intrinsics = np.array([[focal, 0, W / 2.0], [0, focal, H / 2.0], [0, 0, 1.0]])  # fmt: skip
    return Camera(intrinsics, np.zeros((4, 1)))


def solve_pose_from_nocs(nocs, mask, extents, camera, max_points=1500, seed=0):
    rows, cols = np.nonzero(mask)
    if len(rows) < 4:
        return None
    points2D = np.stack([cols, rows], axis=1).astype("float64")
    points3D = extents * (nocs[rows, cols] - 0.5)
    if len(points3D) > max_points:
        choice = np.random.RandomState(seed).choice(len(points3D), max_points, False)  # fmt: skip
        points2D, points3D = points2D[choice], points3D[choice]
    pose6D = solve_PnP_RANSAC(points2D, points3D, camera)
    if pose6D is None:
        return None
    rotation = cv2.Rodrigues(pose6D.rotation_vector)[0]
    return rotation, np.asarray(pose6D.translation).reshape(3)
