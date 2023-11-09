import numpy as np

from paz.backend.image.opencv_image import resize_image
from paz.backend.image.image import cast_image


def subtract_mean_image(image, mean_pixel_values):
    """Expects an RGB image (or array of images) and subtracts
    the mean pixel and converts it to float. Expects image
    colors in RGB order.

    # Arguments:
        image: [height, width, channel] RGB image of type int.
        pixel_values: array of mean rgb pixel value.

    # Returns:
        Normalised image of type float32.
    """
    return image - mean_pixel_values


def add_mean_image(normalized_image, mean_pixel_values):
    """Takes a image normalized and returns the original.

    # Arguments:
        normalised_image: [height, width, channel] normalised RGB image of type
                           float.
        pixel_values: array of mean rgb pixel value.

    # Returns:
        RGB image values of type int.
    """
    return normalized_image + mean_pixel_values


def crop_resize_masks(boxes, mask, small_mask_shape):
    """Resize masks to a smaller version to reduce memory load.

    # Arguments:
        boxes: [instances, ymin, xmin, ymax, xmax] Bounding box.
        mask: [height, width, channel] Binary mask.
        small_mask_shape : [height, width] shape of the small mask to be
                           converted.

    # Returns:
        smaller_masks: [height, width, channel]

    """
    smaller_masks = np.zeros(small_mask_shape + (mask.shape[-1],), dtype=bool)
    for instance in range(mask.shape[-1]):
        small_mask = mask[:, :, instance]
        y1, x1, y2, x2 = boxes[instance, :4]
        small_mask = small_mask[y1:y2, x1:x2]

        small_mask = resize_image(np.array(small_mask), small_mask_shape)
        smaller_masks[:, :, instance] = np.around(small_mask).astype(bool)
    return smaller_masks


def resize_to_original_size(mask, box, image_shape, threshold=0.5):
    """Create masks back to original shape from smaller masks.

    # Arguments:
        mask: [height, width] of type float. Typically 28x28 mask.
        box: [y_min, x_min, y_max, x_max]. The box to fit the mask in.
        image_shape: [height, width, channel].

    # Returns:
        A binary mask with the same size as the original image.
    """
    box = [int(x) for x in box]
    y_min, x_min, y_max, x_max = box
    mask = resize_image(mask, (int(x_max - x_min), int(y_max - y_min)))

    mask = np.where(mask >= threshold, 1, 0).astype(np.bool)
    full_mask = np.zeros(image_shape[:2], dtype=np.bool)
    full_mask[int(y_min):int(y_max), int(x_min):int(x_max)] = mask
    return full_mask
