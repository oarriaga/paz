"""High-level Depth Anything 3 estimators.

Preprocessing happens outside the compiled model: images are resized to a
fixed square resolution (a multiple of the patch size), scaled to [0, 1], and
ImageNet-normalized. ``weights="pretrained"`` (the default) downloads the
hosted weights; pass a local ``.weights.h5`` path to override.

Any-view estimators return, in order:
``depth, depth_confidence, extrinsics, intrinsics, rays, ray_confidence``.
Monocular estimators return ``depth, sky``; the metric estimator returns
``depth_meters, sky``.
"""
import numpy as np
from keras import ops

from paz.backend.image import resize_opencv, standardize
from paz.models import DepthAnything3Small, DepthAnything3Base
from paz.models import DepthAnything3MonoLarge, DepthAnything3MetricLarge
from paz.models.foundation.depth_anything3 import pretrained


def EstimateDepthAnything3Small(weights="pretrained", image_size=518):
    path = resolve_weights(weights, "da3_small")
    args = DepthAnything3Small, path, image_size
    return build_any_view_estimator(*args)


def EstimateDepthAnything3Base(weights="pretrained", image_size=518):
    path = resolve_weights(weights, "da3_base")
    args = DepthAnything3Base, path, image_size
    return build_any_view_estimator(*args)


def build_any_view_estimator(builder, weights_path, image_size):
    cache = {}

    def estimate(images):
        views = preprocess_views(images, image_size)
        args = cache, builder, views.shape[1], image_size, weights_path
        return load_view_model(*args)(views)

    return estimate


def EstimateDepthAnything3MonoLarge(weights="pretrained", image_size=518):
    path = resolve_weights(weights, "da3_mono_large")
    model = load_mono_model(DepthAnything3MonoLarge, path, image_size)

    def estimate(image):
        return model(preprocess_batch(image, image_size))

    return estimate


def EstimateDepthAnything3MetricLarge(focal_length, weights="pretrained",
                                      image_size=518):
    path = resolve_weights(weights, "da3_metric_large")
    model = load_mono_model(DepthAnything3MetricLarge, path, image_size)

    def estimate(image):
        depth, sky = model(preprocess_batch(image, image_size))
        return focal_length * depth / 300.0, sky

    return estimate


def resolve_weights(weights, model_name):
    if weights == "pretrained":
        return pretrained.download_weights(model_name)
    return weights


def load_mono_model(builder, weights_path, image_size):
    model = builder((image_size, image_size, 3))
    model.load_weights(weights_path)
    return model


def load_view_model(cache, builder, num_views, image_size, weights_path):
    if num_views not in cache:
        model = builder(num_views, (image_size, image_size, 3))
        model.load_weights(weights_path)
        cache[num_views] = model
    return cache[num_views]


def preprocess_views(images, image_size):
    views = [preprocess_image(image, image_size) for image in images]
    return ops.expand_dims(ops.stack(views, axis=0), axis=0)


def preprocess_batch(image, image_size):
    return ops.expand_dims(preprocess_image(image, image_size), axis=0)


def preprocess_image(image, image_size):
    mean = np.array([0.485, 0.456, 0.406], "float32")
    std = np.array([0.229, 0.224, 0.225], "float32")
    array = ops.convert_to_tensor(to_float(image))
    resized = resize_opencv(array, (image_size, image_size))
    return standardize(resized, mean, std)


def to_float(image):
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image.astype("float32") / 255.0
    return image.astype("float32")
