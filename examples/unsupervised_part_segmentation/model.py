import numpy as np
import paz
from dino import compute_features
from pca import apply_PCA, apply_masked_PCA


def compute_aspect_shape(H, W, smaller_edge_size):
    # assumes H is the smaller edge size
    W = int((W / H) * smaller_edge_size)
    return smaller_edge_size, W


def preprocess(image, crop_shape, mean, stdv):
    size, size = crop_shape
    H, W = compute_aspect_shape(*paz.image.get_size(image), size)
    image = paz.image.resize(image, (H, W))
    image = paz.image.crop_center(image, *crop_shape)
    image = paz.image.normalize(image)
    image = paz.image.standardize(image, mean, stdv)
    return image


def postprocess_features(projected_features, foreground_masks):
    foreground_masks = np.squeeze(foreground_masks, axis=-1)
    foreground_features = projected_features[foreground_masks]
    foreground_features = paz.image.normalize_min_max(foreground_features)
    image = np.zeros_like(projected_features)
    image[foreground_masks] = foreground_features
    return image


def compute_grid_size(H_cropped, W_cropped, patch_size):
    grid_size = (H_cropped // patch_size, W_cropped // patch_size)
    return grid_size


def features_to_grid(features, crop_shape, patch_size, image_size):
    grid_size = compute_grid_size(*crop_shape, patch_size)
    # patch_size might be a tuple? DINOv3 model has int patch_size usually (16).
    if isinstance(patch_size, tuple):
        patch_size = patch_size[0]
    return np.reshape(features, (image_size, *grid_size, -1))


def compute_joint_features(model, images, mean, stdv, crop_shape, patch_size):
    joint_features = []
    for image in images:
        features = compute_features(
            model, preprocess(image, crop_shape, mean, stdv)
        )
        joint_features.append(features)
    joint_features = np.concatenate(joint_features)
    return features_to_grid(joint_features, crop_shape, patch_size, len(images))


def compute_foreground_masks(
    model, size, patch_size, mean, stdv, threshold, images
):
    joint_features = compute_joint_features(
        model, images, mean, stdv, size, patch_size
    )
    joint_features = apply_PCA(joint_features, 3)
    joint_features = paz.image.normalize_min_max(joint_features)
    foreground_masks = joint_features[..., 0:1] <= threshold
    # foreground_masks = joint_features[..., 0:1] > threshold

    return foreground_masks


def project_features(dimension, masks, joint_features):
    features = apply_masked_PCA(joint_features, masks, dimension)
    return postprocess_features(features, masks)
