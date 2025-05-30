import jax.numpy as jp
import paz


def draw_boxes_and_points(image, boxes, all_points, box_color, points_colors):
    image = paz.draw.boxes(image, boxes, box_color, thickness=3)
    for points in all_points:
        image = paz.draw.keypoints(image, points, points_colors, 8)
    return image


def denormalize_keypoints(keypoints, height, width):
    """Transform normalized keypoint coordinates into image coordinates
    # Arguments
        keypoints: Numpy array of shape ``(num_keypoints, 2)``.
        height: Int. Height of the image
        width: Int. Width of the image
    # Returns
        Numpy array of shape ``(num_keypoints, 2)``.
    """
    for keypoint_arg, keypoint in enumerate(keypoints):
        x, y = keypoint[:2]
        # transform key-point coordinates to image coordinates
        x = (min(max(x, -1), 1) * width / 2 + width / 2) - 0.5
        # flip since the image coordinates for y are flipped
        y = height - 0.5 - (min(max(y, -1), 1) * height / 2 + height / 2)
        x, y = int(round(x)), int(round(y))
        keypoints = keypoints.at[keypoint_arg, :2].set([x, y])
    return keypoints.astype(jp.int32)


def denormalize(keypoints, H, W):
    """Transform normalized keypoint coordinates into image coordinates.

    # Arguments
        keypoints: Array of shape ``(num_keypoints, 2)``.
                   Normalized coordinates are expected in the range [-1.0, 1.0].
        height: Int. Height of the image.
        width: Int. Width of the image.

    # Returns
        Array of shape ``(num_keypoints, 2)`` with integer pixel coordinates.
    """
    keypoints = jp.clip(keypoints, -1.0, 1.0)
    keypoints = keypoints + 1.0
    keypoints = keypoints / 2.0
    x, y = paz.points2D.split(keypoints)
    x = (W - 1.0) * x
    y = (H - 1.0) * (1.0 - y)
    denormalized_keypoints = paz.points2D.merge(x, y)
    denormalized_keypoints = jp.round(denormalized_keypoints)
    return paz.cast(denormalized_keypoints, "int32")


def DetectFaceKeypointNet2D32(box_scale=1.2, draw=None):
    detect = paz.models.HaarCascadeFrontalFaceDetector(draw=False)
    estimate_keypoints = FaceKeypointNet2D32(draw=False)
    colors = paz.draw.lincolor(15 + 1)
    if draw is None:
        draw = paz.lock(draw_boxes_and_points, colors[-1], colors[:-1])

    def call(image):
        boxes = paz.detection.get_boxes(detect(image))
        boxes = paz.boxes.square(boxes)
        boxes = paz.boxes.scale(boxes, box_scale, box_scale)
        boxes = paz.cast(boxes, "int32")
        boxes = paz.boxes.remove_invalid(boxes)
        total_keypoints = []
        for box in boxes:
            keypoints = estimate_keypoints(paz.image.crop(image, box))
            keypoints = paz.points2D.shift_to_box_origin(keypoints, box)
            total_keypoints.append(keypoints)
        total_keypoints = jp.array(total_keypoints)
        return boxes, total_keypoints

    return (lambda x: (y := call(x), draw(x, *y))) if callable(draw) else call


def FaceKeypointNet2D32(draw=None):
    model = paz.models.FaceKeypointNet2D32()
    if draw is None:
        draw = paz.lock(paz.draw.keypoints, paz.draw.lincolor(15), 3)

    def preprocess(image):
        image = paz.image.resize_opencv(image, paz.image.get_input_size(model))
        image = paz.image.RGB_to_GRAY(image)
        image = paz.image.normalize(image)
        return jp.expand_dims(image, axis=[0, -1])

    def postprocess(keypoints, H, W):
        keypoints = jp.squeeze(keypoints, axis=0)
        # keypoints = paz.points2D.denormalize(keypoints, H, W)
        keypoints = denormalize(keypoints, H, W)
        # keypoints = denormalize_keypoints(keypoints, H, W)
        return keypoints

    def call(image):
        return postprocess(model(preprocess(image)), *paz.image.get_size(image))

    return (lambda x: (y := call(x), draw(x, y))) if callable(draw) else call
