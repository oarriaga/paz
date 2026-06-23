from collections import namedtuple

import jax.numpy as jp
import numpy as np
import paz
from paz.datasets.hands import MPIIHandJoints
from paz.datasets.hands import hand_links, link_colors, joint_colors


HandPose = namedtuple(
    "HandPose",
    ["keypoints2D", "keypoints3D", "absolute_angles", "relative_angles"],
)


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


def draw_hand_skeleton(image, keypoints2D, link_width=2, radius=4):
    image = np.array(image)
    points = [(int(x), int(y)) for x, y in keypoints2D]
    for link_arg, (parent, child) in enumerate(hand_links):
        color = link_colors[link_arg].tolist()
        image = paz.draw.line(image, points[parent], points[child], color, link_width)  # fmt: skip
    for point_arg, point in enumerate(points):
        image = paz.draw.keypoint(image, point, joint_colors[point_arg].tolist(), radius)  # fmt: skip
    return image


def DetNetHandKeypoints(right_hand=False, draw=None, grid=32, input_size=128):
    model = paz.models.DetNet(weights="detnet")
    if draw is None:
        draw = draw_hand_skeleton

    def preprocess(image):
        image = paz.image.resize_opencv(image, (input_size, input_size))
        image = image[:, ::-1] if right_hand else image
        return jp.expand_dims(image, axis=0)

    def postprocess(keypoints3D, uv, H, W):
        keypoints3D = paz.to_numpy(keypoints3D)
        uv = paz.to_numpy(uv)
        if right_hand:
            uv[:, 0] = grid - uv[:, 0]
        keypoints2D = uv[:, ::-1]
        scale = np.array([W / input_size * 4, H / input_size * 4])
        keypoints2D = (keypoints2D * scale).astype(np.int32)
        return keypoints2D, keypoints3D

    def call(image):
        keypoints3D, uv = model(preprocess(image))
        return postprocess(keypoints3D, uv, *paz.image.get_size(image))

    return (lambda x: (y := call(x), draw(x, y[0]))) if callable(draw) else call


def IKNetHandJointAngles(right_hand=False):
    model = paz.models.IKNet(weights="iknet")
    parents = MPIIHandJoints.parents
    links_origin = MPIIHandJoints.links_origin
    if right_hand:
        links_origin = paz.angles.flip_along_x_axis(links_origin)
    links_delta = paz.angles.compute_orientation_vector(links_origin, parents)

    def call(keypoints3D):
        delta = paz.angles.compute_orientation_vector(keypoints3D, parents)
        pack = [keypoints3D, delta, links_origin, links_delta]
        pack = np.concatenate(pack, axis=0)
        absolute_angles = model(jp.expand_dims(pack, axis=0))
        absolute_angles = paz.to_numpy(jp.squeeze(absolute_angles, axis=0))
        relative_angles = paz.angles.compute_relative_angles(
            absolute_angles, right_hand)
        return absolute_angles, relative_angles

    return call


def MinimalHandPoseEstimation(right_hand=False, draw=None):
    estimate_keypoints = DetNetHandKeypoints(right_hand=right_hand, draw=False)
    estimate_angles = IKNetHandJointAngles(right_hand=right_hand)
    if draw is None:
        draw = draw_hand_skeleton

    def call(image):
        keypoints2D, keypoints3D = estimate_keypoints(image)
        absolute_angles, relative_angles = estimate_angles(keypoints3D)
        return HandPose(
            keypoints2D, keypoints3D, absolute_angles, relative_angles)

    def call_and_draw(image):
        hand_pose = call(image)
        return hand_pose, draw(image, hand_pose.keypoints2D)

    return call_and_draw if callable(draw) else call


def DetectMinimalHand(box_scale=1.5, right_hand=False, draw=None):
    detect = paz.applications.SSD512HandDetection(draw=False)
    estimate = MinimalHandPoseEstimation(right_hand=right_hand, draw=False)
    if draw is None:
        draw = draw_hand_skeleton

    def call(image):
        boxes = detect(image)[0]
        boxes = paz.boxes.square(boxes)
        boxes = paz.boxes.scale(boxes, box_scale, box_scale)
        boxes = paz.cast(boxes, "int32")
        boxes = paz.boxes.remove_invalid(boxes)
        all_keypoints2D = []
        for box in boxes:
            keypoints2D = estimate(paz.image.crop(image, box)).keypoints2D
            keypoints2D = paz.points2D.shift_to_box_origin(keypoints2D, box)
            all_keypoints2D.append(keypoints2D)
        return boxes, all_keypoints2D

    def call_and_draw(image):
        boxes, all_keypoints2D = call(image)
        for keypoints2D in all_keypoints2D:
            image = draw(image, keypoints2D)
        return (boxes, all_keypoints2D), image

    return call_and_draw if callable(draw) else call
