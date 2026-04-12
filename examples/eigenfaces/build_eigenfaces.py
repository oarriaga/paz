import os

import argparse
from pathlib import Path

import jax.numpy as jp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import offsetbox

import eigenfaces
import paz


def load_or_build_state(images, total_variance, experiments_path):
    filepath = Path(experiments_path) / "eigenfaces_state.npz"
    if filepath.exists():
        return eigenfaces.load(filepath)
    eigenface_state = eigenfaces.build(images, total_variance)
    eigenfaces.save(filepath, eigenface_state)
    return eigenface_state


def show_image(image, title):
    plt.figure()
    plt.imshow(np.squeeze(paz.to_numpy(image), axis=-1), cmap="gray")
    plt.title(title)
    plt.axis("off")


def show_eigenvalues(eigenface_state):
    plt.figure()
    plt.plot(paz.to_numpy(eigenface_state.eigenvalues[:30]))
    plt.title("Eigenvalues")


def show_eigenfaces(eigenface_state, num_faces):
    args = (eigenface_state.eigenfaces[:num_faces], eigenface_state.face_shape)
    face_images = display_faces(*args)
    mosaic = paz.draw.mosaic(paz.to_numpy(face_images), (-1, 10))
    show_image(mosaic, f"First {len(face_images)} eigenfaces")


def build_embedding_coordinates(weights):
    if weights.shape[1] >= 2:
        return weights[:, :2]
    zeros = jp.zeros((len(weights), 1), dtype=weights.dtype)
    return jp.concatenate([weights, zeros], axis=1)


def build_thumbnail_images(images):
    thumbnails = []
    for image in images:
        thumbnail = paz.image.resize(image, (20, 20))
        thumbnails.append(paz.cast(thumbnail, jp.uint8))
    return thumbnails


def show_embeddings(images, eigenface_state, num_faces):
    images = images[:num_faces]
    weights = eigenfaces.project(images, eigenface_state)
    thumbnails = build_thumbnail_images(images)
    coordinates = build_embedding_coordinates(weights)
    kwargs = dict(title="Embedding projections", epsilon=0.0)
    plot_embeddings(coordinates, thumbnails, **kwargs)


def display_faces(vectors, face_shape, scale=255.0):
    vectors = jp.array(vectors, dtype=jp.float32)
    face_shape = tuple(int(v) for v in face_shape)
    face_shape = (*face_shape, 1) if len(face_shape) == 2 else face_shape
    images = jp.reshape(vectors, (len(vectors), *face_shape))
    images = paz.image.normalize_min_max(images, axis=(1, 2, 3))
    return paz.cast(scale * images, jp.uint8)


def plot_embeddings(x, images, title=None, epsilon=4e-3, cmap=plt.cm.gray):
    x = np.asarray(paz.to_numpy(x))
    images = [np.asarray(paz.to_numpy(image)) for image in images]
    x_min, x_max = np.min(x, 0), np.max(x, 0)
    x_span = np.where(x_max > x_min, x_max - x_min, 1.0)
    x = (x - x_min) / x_span
    plt.figure()
    axis = plt.subplot(111)
    if hasattr(offsetbox, "AnnotationBbox"):
        _draw_embedding_images(axis, x, images, epsilon, cmap)
    plt.xticks([])
    plt.yticks([])
    if title is not None:
        axis.set_title(title)


def _draw_embedding_images(axis, x, images, epsilon, cmap):
    shown_images = np.array([[1.0, 1.0]])
    for sample, image in zip(x, images):
        distance = np.sum((sample - shown_images) ** 2, axis=1)
        if np.min(distance) < epsilon:
            continue
        shown_images = np.concatenate([shown_images, sample[None]], axis=0)
        box = offsetbox.OffsetImage(image, cmap=cmap)
        imagebox = offsetbox.AnnotationBbox(box, sample)
        axis.add_artist(imagebox)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eigenfaces with JAX PAZ")
    parser.add_argument("--dataset", default="FERPlus")
    parser.add_argument("--split", default="train")
    parser.add_argument("--total_variance", default=0.95, type=float)
    parser.add_argument("--experiments_path", default="experiments")
    parser.add_argument("--num_mosaic_faces", default=80, type=int)
    parser.add_argument("--num_embedding_faces", default=100, type=int)
    args = parser.parse_args()

    images, _ = paz.datasets.load(args.dataset, split=args.split)
    images = jp.array(images, dtype=jp.float32)
    build_args = (images, args.total_variance, args.experiments_path)
    eigenface_state = load_or_build_state(*build_args)
    show_image(eigenfaces.mean_face(eigenface_state), "Mean face")
    show_eigenvalues(eigenface_state)
    show_eigenfaces(eigenface_state, args.num_mosaic_faces)
    show_embeddings(images, eigenface_state, args.num_embedding_faces)
    plt.show()
