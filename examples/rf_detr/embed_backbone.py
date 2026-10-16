"""Projects RF-DETR backbone features to two dimensions.

Taps one layer of a detector, pools it into a vector per image and lays
those out with t-SNE, colored by each image's first VOC class. That shows
what the backbone already separates, and what a fine-tune changes. Any layer
name works: ``projector_norm`` is the map that feeds the decoder, while
``block_5_add2`` sits halfway up the backbone.

Loads the published COCO weights unless --weights names a checkpoint, in
which case --num_classes has to match it. Needs scikit-learn.
"""
import argparse
import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import matplotlib.pyplot as plt
from keras import Model, ops
from sklearn.manifold import TSNE

import paz
from paz.utils import plot
from generator import IMAGENET_MEAN, IMAGENET_STDV

VARIANTS = {
    "nano": paz.models.RFDETRNano,
    "small": paz.models.RFDETRSmall,
    "medium": paz.models.RFDETRMedium,
    "base": paz.models.RFDETRBase,
    "large": paz.models.RFDETRLarge,
}


def build_model(variant, num_classes, weights):
    model = VARIANTS[variant](num_classes)
    if weights is None:
        weights = paz.models.detection.rf_detr.download_weights(model)
    model.load_weights(weights)
    return model


def build_tap(model, layer_name):
    """Model returning one pooled feature vector per image."""
    features = model.get_layer(layer_name).output
    axes = tuple(range(1, len(features.shape) - 1))
    return Model(model.input, ops.mean(features, axis=axes))


def read_features(tap, image_paths, resolution, batch_size):
    """Pools every image through the tap, one batch at a time."""
    features = []
    for start in range(0, len(image_paths), batch_size):
        batch = load_batch(image_paths[start:start + batch_size], resolution)
        features.append(np.asarray(tap.predict(batch, verbose=0)))
    return np.concatenate(features)


def load_batch(image_paths, resolution):
    """Squeezes each image into the detector input and standardizes it."""
    mean = np.asarray(IMAGENET_MEAN, "float32")
    stdv = np.asarray(IMAGENET_STDV, "float32")
    images = []
    for image_path in image_paths:
        image = paz.cast(paz.image.load(image_path), "float32")
        resized = paz.image.resize(image, (resolution, resolution))
        images.append(np.asarray(resized) / 255.0)
    return (np.stack(images) - mean) / stdv


def read_labels(ground_truths):
    """First annotated class of every image, which colors the figure."""
    labels = []
    for ground_truth in ground_truths:
        labels.append(int(np.asarray(ground_truth)[0, 4]))
    return np.asarray(labels)


def project_features(features, seed):
    """t-SNE keeps perplexity below the sample count, so it is derived."""
    perplexity = min(30.0, max(5.0, len(features) / 4.0))
    kwargs = dict(perplexity=perplexity, random_state=seed, init="pca")
    return TSNE(n_components=2, **kwargs).fit_transform(features)


def draw_embedding(points, labels, names, path):
    figure, axis = plt.subplots(figsize=(6.0, 6.0))
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, len(names)))
    for class_arg, class_name in enumerate(names):
        chosen = labels == class_arg
        if chosen.any():
            args = points[chosen, 0], points[chosen, 1]
            kwargs = dict(axis=axis, label=class_name, s=14)
            plot.scatter(*args, color=colors[class_arg], **kwargs)
    plot.set_labels(axis, x="t-SNE 1", y="t-SNE 2")
    plot.clean(axis)
    plot.legend(axis, fontsize=6)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="nano", choices=list(VARIANTS))
    parser.add_argument("--layer", default="projector_norm")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--num_classes", default=91, type=int)
    parser.add_argument("--num_images", default=500, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--seed", default=777, type=int)
    parser.add_argument("--output", default="backbone_features.png")
    args = parser.parse_args()

    model = build_model(args.variant, args.num_classes, args.weights)
    tap = build_tap(model, args.layer)
    images, ground_truths = paz.datasets.voc.load("VOC2007", "test")
    images = images[:args.num_images]
    ground_truths = ground_truths[:args.num_images]
    resolution = model.input_shape[1]
    features = read_features(tap, images, resolution, args.batch_size)
    print("pooled", features.shape, "from", args.layer)
    points = project_features(features, args.seed)
    names = paz.datasets.voc.get_class_names()
    draw_embedding(points, read_labels(ground_truths), names, args.output)
    print("saved", args.output)
