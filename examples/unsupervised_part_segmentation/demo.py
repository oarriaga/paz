import os
import argparse
import paz
import numpy as np
import matplotlib.pyplot as plt
import paz.models

from model import (
    preprocess,
    compute_foreground_masks,
    compute_joint_features,
    project_features,
)
from pca import apply_PCA_K_means, cluster_features


parser = argparse.ArgumentParser(
    description="Unsupervised Segmentation with DINOv3"
)
parser.add_argument("--images_path", default="data/hippogriff", type=str)
parser.add_argument(
    "--H_crop", default=512, type=int
)  # changed to 512 to be div by 16
parser.add_argument("--W_crop", default=512, type=int)
parser.add_argument("--cluster_dimension", default=20, type=int)
parser.add_argument("--num_parts", default=5, type=int)
# parser.add_argument("--threshold", default=0.55, type=float)
parser.add_argument("--threshold", default=0.6, type=float)
parser.add_argument(
    "--dino", default="vitl16", type=str, choices=["vits16", "vitb16", "vitl16"]
)
parser.add_argument("--mean", default=paz.image.rgb_IMAGENET_MEAN)
parser.add_argument("--stdv", default=paz.image.rgb_IMAGENET_STDV)
args = parser.parse_args()
crop_shape = (args.H_crop, args.W_crop)


def validate_crop_shape(crop_shape, patch_size):
    assert (
        crop_shape[0] % patch_size == 0
    ), f"Height {crop_shape[0]} not divisible by patch size {patch_size}"
    assert (
        crop_shape[1] % patch_size == 0
    ), f"Width {crop_shape[1]} not divisible by patch size {patch_size}"


if not os.path.exists(args.images_path):
    print(f"Directory {args.images_path} not found. Please provide valid path.")
    exit(1)

images, grid_images = [], []
image_filenames = sorted(os.listdir(args.images_path))
if not image_filenames:
    print(f"No images found in {args.images_path}")
    exit(1)

for filename in image_filenames:
    if not (
        filename.endswith(".png")
        or filename.endswith(".jpg")
        or filename.endswith(".jpeg")
    ):
        continue
    image = paz.image.load(os.path.join(args.images_path, filename))
    images.append(image)
    grid_images.append(preprocess(image, crop_shape, 0, 1))

if not images:
    print("No valid images loaded.")
    exit(1)

mosaic = paz.draw.mosaic(np.array(grid_images), border=5, background=0.0)
paz.image.show(paz.image.denormalize(mosaic))

print(f"Loading model: {args.dino}")
model_mapping = {
    "vits16": paz.models.DINOV3VITS,
    "vitb16": paz.models.DINOV3VITB,
    "vitl16": paz.models.DINOV3VITL,
}

if args.dino not in model_mapping:
    raise ValueError(f"Unknown model: {args.dino}")

# Instantiate model
# input_shape should match crop_shape + (3,)
model = model_mapping[args.dino](input_shape=(args.H_crop, args.W_crop, 3))

# Keras model patch size
patch_size = model.patch_size
# patch_size in DINOv3 implementation is a tuple (16, 16) or int?
# In vision_transformer.py: self.patch_size = patch_size (argument)
# But PatchEmbed uses make_2tuple.
# Let's check.
if isinstance(patch_size, tuple):
    patch_size = patch_size[0]

validate_crop_shape(crop_shape, patch_size)

print("Computing foreground masks...")
model_args = (model, crop_shape, patch_size, args.mean, args.stdv)
masks = compute_foreground_masks(*model_args, args.threshold, images)

print("Computing joint features...")
joint_features = compute_joint_features(
    model, images, args.mean, args.stdv, crop_shape, patch_size
)

print("Projecting features...")
features = project_features(3, masks, joint_features)

plt.imshow(paz.draw.mosaic(masks, (2, 2), 2, 0))
plt.title("Foreground Masks")
plt.show()

plt.imshow(paz.draw.mosaic(features, (2, 2), 2, 0.0))
plt.title("Projected Features")
plt.show()

print("Clustering features...")
cluster = paz.partial(apply_PCA_K_means, args.cluster_dimension)
clusters = cluster_features(cluster, args.num_parts, joint_features, masks)

plt.imshow(
    paz.draw.mosaic(clusters, (2, 2), 2, background=0), cmap=plt.cm.Paired
)
plt.title("Clusters")
plt.show()
