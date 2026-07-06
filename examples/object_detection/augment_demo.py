import os

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jp
import numpy as np
import paz

SIZE = 300
num_samples = 4
num_augmented = 4


def normalize_boxes(detection, H, W):
    detection = np.asarray(detection, "float32").copy()
    detection[:, :4] = detection[:, :4] / np.array([W, H, W, H])
    return detection


def draw_normalized_boxes(image, detection):
    """Draws normalized corner boxes on an RGB image for visualization."""
    image = np.ascontiguousarray(np.asarray(image, "uint8"))
    valid = detection[detection[:, 4] >= 0.0]
    boxes = valid[:, :4] * np.array([SIZE, SIZE, SIZE, SIZE])
    return paz.draw.boxes(image, boxes.astype("int32"))


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    mean = jp.asarray(paz.image.BGR_IMAGENET_MEAN, jp.float32)
    image_paths, detections = paz.datasets.load("VOC2007", "trainval")

    rows = []
    for sample_arg in range(num_samples):
        raw_image = paz.image.load(image_paths[sample_arg])
        H, W = paz.image.get_size(raw_image)
        image = paz.image.resize(raw_image, (SIZE, SIZE))
        detection = normalize_boxes(detections[sample_arg], H, W)
        panels = [draw_normalized_boxes(image, detection)]
        for augment_arg in range(num_augmented):
            key, sample_key = jax.random.split(key)
            args = sample_key, image, jp.asarray(detection), mean
            augmented_image, augmented = paz.detection.augment_detection(*args)
            augmented = np.asarray(augmented)
            panels.append(draw_normalized_boxes(augmented_image, augmented))
        rows.append(np.hstack(panels))

    montage = np.asarray(paz.image.RGB_to_BGR(np.vstack(rows)))
    paz.image.write("augmentations.jpg", montage)
    print("wrote augmentations.jpg: left column is the original, then %d "
          "random augmentations per row" % num_augmented)
