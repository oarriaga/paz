import jax.numpy as jp
import paz


def DetectFaceKeypointNet2D32(box_scale=1.2, draw=None):
    detect = paz.models.HaarCascadeFrontalFaceDetector(draw=False)
    estimate_keypoints = FaceKeypointNet2D32(draw=False)
    colors = paz.draw.lincolor(15 + 1)
    if draw is None:
        draw = paz.lock(paz.draw.boxes_and_points, colors[-1], colors[:-1])

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
        keypoints = paz.points2D.denormalize(keypoints, H, W)
        return keypoints

    def call(image):
        return postprocess(model(preprocess(image)), *paz.image.get_size(image))

    return (lambda x: (y := call(x), draw(x, y))) if callable(draw) else call
