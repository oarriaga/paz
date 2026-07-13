"""High-level Depth Anything 3 estimators.

Preprocessing happens outside the compiled model: images are resized to a
fixed square resolution (a multiple of the patch size), scaled to [0, 1], and
ImageNet-normalized. Weights come from a converted ``.weights.h5`` file; pass
its path explicitly until hosting is available.

Any-view estimators return, in order:
``depth, depth_confidence, extrinsics, intrinsics, rays, ray_confidence``.
Monocular estimators return ``depth, sky``; the metric estimator returns
``depth_meters, sky``.
"""
import cv2
import numpy as np
from keras import ops

from paz.models.foundation.depth_anything3.models import build_da3_small
from paz.models.foundation.depth_anything3.models import build_da3_base
from paz.models.foundation.depth_anything3.models import build_da3_mono_large
from paz.models.foundation.depth_anything3.models import build_da3_metric_large

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], "float32")
IMAGENET_STD = np.array([0.229, 0.224, 0.225], "float32")
METRIC_DIVISOR = 300.0


def EstimateDepthAnything3Small(weights_path, image_size=518):
    return build_any_view_estimator(build_da3_small, weights_path, image_size)


def EstimateDepthAnything3Base(weights_path, image_size=518):
    return build_any_view_estimator(build_da3_base, weights_path, image_size)


def build_any_view_estimator(builder, weights_path, image_size):
    models = {}

    def estimate(images):
        views = preprocess_views(images, image_size)
        model = load_for_views(models, builder, views.shape[1], image_size,
                               weights_path)
        depth, confidence, extrinsics, intrinsics, rays, ray_confidence = model(views)
        return depth, confidence, extrinsics, intrinsics, rays, ray_confidence

    return estimate


def EstimateDepthAnything3MonoLarge(weights_path, image_size=518):
    model = build_mono_model(build_da3_mono_large, weights_path, image_size)

    def estimate(image):
        depth, sky = model(preprocess_batch(image, image_size))
        return depth, sky

    return estimate


def EstimateDepthAnything3MetricLarge(weights_path, focal_length, image_size=518):
    model = build_mono_model(build_da3_metric_large, weights_path, image_size)

    def estimate(image):
        depth, sky = model(preprocess_batch(image, image_size))
        return focal_length * depth / METRIC_DIVISOR, sky

    return estimate


def build_mono_model(builder, weights_path, image_size):
    model = builder((image_size, image_size, 3))
    model.load_weights(weights_path)
    return model


def load_for_views(models, builder, num_views, image_size, weights_path):
    if num_views not in models:
        model = builder(num_views, (image_size, image_size, 3))
        model.load_weights(weights_path)
        models[num_views] = model
    return models[num_views]


def preprocess_views(images, image_size):
    views = [preprocess_image(image, image_size) for image in images]
    return ops.expand_dims(ops.stack(views, axis=0), axis=0)


def preprocess_batch(image, image_size):
    return ops.expand_dims(preprocess_image(image, image_size), axis=0)


def preprocess_image(image, image_size):
    image = np.asarray(image)
    if image.dtype == np.uint8:
        image = image.astype("float32") / 255.0
    resized = cv2.resize(image.astype("float32"), (image_size, image_size),
                         interpolation=cv2.INTER_AREA)
    normalized = (resized - IMAGENET_MEAN) / IMAGENET_STD
    return ops.convert_to_tensor(normalized, dtype="float32")
