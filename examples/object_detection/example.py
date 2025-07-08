import jax
import jax.numpy as jp
import paz

image = paz.image.load("photo_2.jpg")
H, W = paz.image.get_size(image)
key = jax.random.PRNGKey(777)
boxes = paz.boxes.sample(key, H, W, 0.8, 0.1, 1)
image_with_boxes = paz.draw.boxes(image, boxes)
paz.image.show(image_with_boxes)
print(boxes.shape)


def validate_aspect_ratio(crop_box, min_aspect_ratio=0.5, max_aspect_ratio=2.0):
    aspect_ratio = paz.boxes.compute_aspect_ratios(crop_box, keepdims=False)
    aspect_ratio = jp.squeeze(aspect_ratio)
    is_valid_min_aspect_ratio = aspect_ratio >= min_aspect_ratio
    is_valid_max_aspect_ratio = aspect_ratio <= max_aspect_ratio
    return is_valid_min_aspect_ratio & is_valid_max_aspect_ratio


def validate_IOU(crop_box, boxes, min_IOU=0.5, max_IOU=0.9):
    IOUs = paz.boxes.compute_IOUs(crop_box, boxes)[0]
    is_valid_min_IOU = IOUs.max() >= min_IOU
    is_valid_max_IOU = IOUs.min() <= max_IOU
    return is_valid_min_IOU & is_valid_max_IOU


def compute_centers_mask(crop_box, boxes):
    x_min, y_min, x_max, y_max = paz.boxes.split(crop_box, False)
    x_centers, y_centers = paz.boxes.compute_centers(boxes)
    centers_above_x_min = x_min < x_centers
    centers_above_y_min = y_min < y_centers
    centers_below_x_max = x_max > x_centers
    centers_below_y_max = y_max > y_centers
    centers_within_x_crop = centers_above_x_min & centers_below_x_max
    centers_within_y_crop = centers_above_y_min & centers_below_y_max
    centers_within_crop = centers_within_x_crop & centers_within_y_crop
    return centers_within_crop


def validate_centers(crop_box, boxes):
    centers_within_crop = compute_centers_mask(crop_box, boxes)
    at_least_one_center_within_crop = centers_within_crop.any()
    return at_least_one_center_within_crop


def is_valid(crop_box, boxes):
    valid_ratio = validate_aspect_ratio(crop_box)
    valid_IOU = validate_IOU(crop_box, boxes)
    valid_centers = validate_centers(crop_box, boxes)
    valid_crop = valid_ratio & valid_IOU & valid_centers
    return valid_crop


def fit_to_crop(boxes, crop_box):
    x_min_crop, y_min_crop, x_max_crop, y_max_crop = crop_box
    x_min, y_min, x_max, y_max = paz.boxes.split(boxes)

    x_min = jp.maximum(x_min, x_min_crop)
    y_min = jp.maximum(y_min, y_min_crop)
    x_max = jp.minimum(x_max, x_max_crop)
    y_max = jp.minimum(y_max, y_max_crop)

    # x_min = x_min - x_min_crop
    # y_min = y_min - y_min_crop
    # x_max = x_max - x_min_crop
    # y_max = y_max - y_min_crop

    return paz.boxes.merge(x_min, y_min, x_max, y_max)


key = jax.random.PRNGKey(777)
key_1, key_2 = jax.random.split(key)
crop_box = paz.boxes.sample(key_1, H, W, 0.8, 0.1, 1)
# boxes = paz.boxes.sample(key_2, H, W, 0.8, 0.1, 5)
# print("STARTING SELECTION")
# boxes = paz.boxes.from_selection(image)
boxes = jp.array(
    [
        [595, 224, 1002, 808],
        [171, 319, 845, 709],
        [594, 458, 1006, 829],
        [1075, 272, 1126, 407],
        [1101, 256, 1156, 400],
        [1209, 277, 1253, 379],
        [1237, 282, 1283, 386],
        [1269, 287, 1311, 385],
        [345, 343, 470, 473],
        [480, 360, 570, 458],
    ],
)


image_with_crop = paz.draw.boxes(image, crop_box, paz.draw.GREEN)
paz.image.show(image_with_crop)
image_with_boxes = paz.draw.boxes(image_with_crop, boxes, paz.draw.RED)
paz.image.show(image_with_boxes)

valid_crop = is_valid(crop_box, boxes)
center_mask = compute_centers_mask(crop_box, boxes)
center_mask = jp.squeeze(center_mask)
new_boxes = fit_to_crop(boxes[center_mask], crop_box[0])
paz.image.show(paz.draw.boxes(image_with_crop, new_boxes, paz.draw.ORANGE))


detections = jp.array(
    [
        [595, 224, 1002, 808, 0],
        [171, 319, 845, 709, 1],
        [594, 458, 1006, 829, 2],
        [1075, 272, 1126, 407, 3],
        [1101, 256, 1156, 400, 4],
        [1209, 277, 1253, 379, 5],
        [1237, 282, 1283, 386, 6],
        [1269, 287, 1311, 385, 7],
        [345, 343, 470, 473, 8],
        [480, 360, 570, 458, 9],
    ],
)


for key in jax.random.split(key, 100):
    new_image, new_dets = paz.scene.random_crop(key, image, detections, 1.0)
    paz.image.show(paz.draw.boxes(new_image, paz.detection.get_boxes(new_dets)))
