"""DeepSORT appearance embedder: MobileNetV2 features of box crops.

Takes RGB frames (the paz convention) and ``[left, top, width, height]``
boxes, and returns one raw 1280-d feature per box. Behaviour was checked
against deep-sort-realtime 1.3.2, used only as a test oracle.
"""
from functools import partial

import jax
import jax.numpy as jp

import paz


def build_embedder(model):
    return partial(embed, build_forward(model))


def embed(forward, image_RGB, boxes2D):
    if len(boxes2D) == 0:
        features = jp.zeros((0, 1280), jp.float32)
    else:
        features = embed_boxes(forward, image_RGB, boxes2D)
    return features


def embed_boxes(forward, image_RGB, boxes2D):
    crops = crop_boxes(image_RGB, boxes2D)
    return compute_features(forward, preprocess_crops(crops))


def crop_boxes(image_RGB, boxes2D):
    crops = []
    for box2D in boxes2D:
        crops.append(crop_box(image_RGB, box2D))
    return crops


def crop_box(image_RGB, box2D):
    x_min, y_min, x_max, y_max = clip_box(image_RGB.shape, box2D)
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"box {box2D} gives an empty crop")
    return image_RGB[y_min:y_max, x_min:x_max]


def clip_box(shape, box2D):
    # int() truncates towards zero, as the reference crops do
    left, top, width, height = [int(value) for value in box2D]
    x_max = min(shape[1], left + width)
    y_max = min(shape[0], top + height)
    return max(0, left), max(0, top), x_max, y_max


def preprocess_crops(crops):
    images = []
    for crop in crops:
        images.append(paz.image.resize_opencv(crop, (224, 224)))
    images = paz.image.normalize(jp.stack(images).astype(jp.float32))
    mean = jp.array([0.485, 0.456, 0.406])
    stdv = jp.array([0.229, 0.224, 0.225])
    return paz.image.standardize(images, mean, stdv)


def build_forward(model):
    apply = jax.jit(partial(apply_model, model))

    def forward(images):
        trainable = read_values(model.trainable_variables)
        non_trainable = read_values(model.non_trainable_variables)
        return apply(trainable, non_trainable, images)

    # the compile count of the jitted call, read by the tests
    forward._cache_size = apply._cache_size
    return forward


def apply_model(model, trainable, non_trainable, images):
    # Ampere GPUs may run float32 convolutions as TF32, which shifts
    # features near the 0.2 cosine gate; keep full float32
    with jax.default_matmul_precision("highest"):
        args = (trainable, non_trainable, images)
        features, _ = model.stateless_call(*args, training=False)
    return features


def read_values(variables):
    return [variable.value for variable in variables]


def compute_features(forward, images):
    features = [forward(batch) for batch in split_batches(images)]
    return jp.concatenate(features)[:len(images)]


def split_batches(images):
    # a fixed batch shape compiles the forward pass once for any count
    batch_size = 16
    num_missing = -len(images) % batch_size
    padding = jp.zeros((num_missing, *images.shape[1:]), images.dtype)
    padded = jp.concatenate([images, padding])
    return padded.reshape(-1, batch_size, *images.shape[1:])
