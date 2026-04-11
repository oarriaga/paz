from collections import namedtuple
from pathlib import Path

import jax.numpy as jp
import numpy as np

import paz

_fields = ["mean_vector", "eigenfaces", "eigenvalues", "face_shape"]
EigenfaceState = namedtuple("EigenfaceState", _fields)


def build(images, total_variance=0.95):
    face_vectors = _build_face_vectors(images)
    pca_state = paz.PCA.fit(face_vectors, min(face_vectors.shape))
    eigenvalues = _compute_eigenvalues(pca_state.variances, len(face_vectors))
    num_components = _compute_num_components(eigenvalues, total_variance)
    face_shape = _normalize_shape(images.shape[1:])
    return _pack_state(pca_state, eigenvalues, num_components, face_shape)


def project(images, state):
    face_vectors = _build_face_vectors(images)
    return paz.PCA.transform(face_vectors, _build_pca_state(state))


def project_single(image, state, detect_face=None):
    face_vector = preprocess(image, state.face_shape, detect_face)
    face_vector = jp.expand_dims(face_vector, axis=0)
    weights = paz.PCA.transform(face_vector, _build_pca_state(state))
    return jp.squeeze(weights, axis=0)


def preprocess(image, face_shape, detect_face=None):
    face_shape = _normalize_shape(face_shape)
    image = _crop_single_face(image, detect_face) if detect_face else image
    image = jp.array(image)
    image_3d = jp.expand_dims(image, axis=-1) if image.ndim == 2 else image
    is_rgb = image_3d.shape[-1] == 3
    image = paz.image.rgb_to_gray(image_3d) if is_rgb else image_3d
    image = paz.image.resize(image, face_shape[:2])
    return jp.reshape(jp.array(image, dtype=jp.float32), (-1,))


def save(filepath, state):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        mean_vector=paz.to_numpy(state.mean_vector),
        eigenfaces=paz.to_numpy(state.eigenfaces),
        eigenvalues=paz.to_numpy(state.eigenvalues),
        face_shape=np.array(state.face_shape, dtype=np.int32),
    )
    np.savez(filepath, **kwargs)


def load(filepath):
    data = np.load(Path(filepath))
    mean = jp.array(data["mean_vector"], dtype=jp.float32)
    faces = jp.array(data["eigenfaces"], dtype=jp.float32)
    values = jp.array(data["eigenvalues"], dtype=jp.float32)
    shape = tuple(int(dim) for dim in data["face_shape"])
    return EigenfaceState(mean, faces, values, shape)


def mean_face(state):
    face_shape = _normalize_shape(state.face_shape)
    face = jp.reshape(jp.array(state.mean_vector), face_shape)
    face = paz.image.normalize_min_max(face, axis=None)
    return paz.cast(255.0 * face, jp.uint8)


def _pack_state(pca_state, eigenvalues, num_components, face_shape):
    mean_vector = jp.array(pca_state.mean, dtype=jp.float32)
    eigenfaces_arr = jp.array(pca_state.base[:num_components], dtype=jp.float32)
    eigenvalues_arr = jp.array(eigenvalues[:num_components], dtype=jp.float32)
    args = (mean_vector, eigenfaces_arr, eigenvalues_arr, tuple(face_shape))
    return EigenfaceState(*args)


def _build_pca_state(state):
    keys = ["mean", "variances", "base"]
    values = [state.mean_vector, state.eigenvalues, state.eigenfaces]
    data = dict(zip(keys, values))
    return paz.NamedTuple("PCAState", **data)


def _build_face_vectors(images):
    images = jp.array(images, dtype=jp.float32)
    images = jp.expand_dims(images, axis=0) if images.ndim == 3 else images
    return jp.reshape(images, (images.shape[0], -1))


def _crop_single_face(image, detect_face):
    boxes = paz.detection.get_boxes(detect_face(image))
    boxes = paz.boxes.remove_invalid(boxes)
    if len(boxes) != 1:
        return image
    return paz.image.crop(image, jp.array(boxes[0], dtype=jp.int32))


def _normalize_shape(face_shape):
    face_shape = tuple(int(dim) for dim in face_shape)
    return (*face_shape, 1) if len(face_shape) == 2 else face_shape


def _compute_eigenvalues(singular_values, num_samples):
    denominator = max(num_samples - 1, 1)
    singular_values = jp.array(singular_values, dtype=jp.float32)
    return (singular_values**2) / denominator


def _compute_num_components(eigenvalues, total_variance):
    if not (0 < total_variance <= 1.0):
        raise ValueError("Variance must be in (0, 1]")
    normalized = eigenvalues / jp.sum(eigenvalues)
    cumulative = jp.cumsum(normalized)
    num_components = int(jp.sum(cumulative < total_variance)) + 1
    return min(num_components, len(eigenvalues))
