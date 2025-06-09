import os

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".50"
# os.environ["KERAS_BACKEND"] = "jax"

import argparse
import paz
import numpy as np
import matplotlib.pyplot as plt
import torch

from model import (
    preprocess,
    compute_foreground_masks,
    compute_joint_features,
    project_features,
)
from pca import apply_PCA_K_means, cluster_features


parser = argparse.ArgumentParser(description="Unsupervised Segmentation")
parser.add_argument("--images_path", default="data/hippogriff", type=str)
parser.add_argument("--H_crop", default=518, type=int)
parser.add_argument("--W_crop", default=518, type=int)
parser.add_argument("--cluster_dimension", default=20, type=int)
parser.add_argument("--num_parts", default=5, type=int)
# parser.add_argument("--min_parts", default=2, type=int)
# parser.add_argument("--max_parts", default=9, type=int)
parser.add_argument("--threshold", default=0.55, type=float)
# parser.add_argument("--dino", default="vitl14_reg", type=str)
parser.add_argument("--dino", default="vitl14", type=str)
parser.add_argument("--mean", default=paz.image.rgb_IMAGENET_MEAN)
parser.add_argument("--stdv", default=paz.image.rgb_IMAGENET_STDV)
args = parser.parse_args()
crop_shape = (args.H_crop, args.W_crop)


def validate_crop_shape(crop_shape, patch_size):
    assert crop_shape[0] % patch_size == 0
    assert crop_shape[1] % patch_size == 0


images, grid_images = [], []
for filename in sorted(os.listdir(args.images_path)):
    image = paz.image.load(os.path.join(args.images_path, filename))
    images.append(image)
    grid_images.append(preprocess(image, crop_shape, 0, 1))
mosaic = paz.draw.mosaic(np.array(grid_images), border=5, background=0.0)
paz.image.show(paz.image.denormalize(mosaic))


model = torch.hub.load("facebookresearch/dinov2", f"dinov2_{args.dino}")
model.to(torch.device("cuda"))
patch_size = model.patch_size
validate_crop_shape(crop_shape, model.patch_size)


model_args = (model, crop_shape, patch_size, args.mean, args.stdv)
masks = compute_foreground_masks(*model_args, args.threshold, images)
joint_features = compute_joint_features(
    model, images, args.mean, args.stdv, crop_shape, patch_size
)
features = project_features(3, masks, joint_features)

plt.imshow(paz.draw.mosaic(masks, (2, 2), 2, 0))
plt.show()

plt.imshow(paz.draw.mosaic(features, (2, 2), 2, 0.0))
plt.show()

cluster = paz.partial(apply_PCA_K_means, args.cluster_dimension)
# fit_scores_args = (joint_features, masks, cluster, min_parts, max_parts)
# fit_scores = compute_fit_scores(*fit_scores_args)
# num_parts = estimate_num_parts(fit_scores)
clusters = cluster_features(cluster, args.num_parts, joint_features, masks)
plt.imshow(
    paz.draw.mosaic(clusters, (2, 2), 2, background=0), cmap=plt.cm.Paired
)
plt.show()
