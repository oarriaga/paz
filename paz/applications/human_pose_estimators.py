import cv2
import numpy as np
import paz
from paz.backend import heatmaps
from paz.datasets.human_pose import COCO_KEYPOINT_ORDER
from paz.datasets.human_pose import human_links, link_colors
from paz.datasets import human36m


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STDV = np.array([0.229, 0.224, 0.225])
NUM_KEYPOINTS = 17


def HigherHRNetHumanPose2D(max_people=30, detection_thresh=0.2,
                           tag_thresh=1.0, draw=None):
    model = paz.models.HigherHRNet(weights="COCO")
    if draw is None:
        draw = draw_human_skeletons

    def preprocess(image):
        image = np.asarray(image)
        center, scale, size = compute_transform_params(image)
        transform = build_affine(center, scale, size, inverse=False)
        warped = cv2.warpAffine(image, transform, tuple(int(v) for v in size))
        normalized = (warped / 255.0 - IMAGENET_MEAN) / IMAGENET_STDV
        return np.expand_dims(normalized, 0).astype("float32"), center, scale

    def postprocess(outputs, center, scale):
        maps, tags = heatmaps.compute_heatmaps_and_tags(outputs, NUM_KEYPOINTS)
        detections = heatmaps.top_k_detections(maps, tags, max_people)
        people = heatmaps.group_by_tag(detections, COCO_KEYPOINT_ORDER,
                                       tag_thresh, detection_thresh)
        if len(people) == 0:
            return [], []
        people = heatmaps.adjust_keypoints(maps[0], people)
        scores = heatmaps.compute_scores(people)
        people = heatmaps.refine_keypoints(maps[0], tags[0], people)
        shape = [maps.shape[3], maps.shape[2]]
        return transform_to_image(people, center, scale, shape), scores

    def call(image):
        model_input, center, scale = preprocess(image)
        people, scores = postprocess(model(model_input), center, scale)
        keypoints = [person[:, :2] for person in people]
        return keypoints, scores

    def call_and_draw(image):
        model_input, center, scale = preprocess(image)
        people, scores = postprocess(model(model_input), center, scale)
        keypoints = [person[:, :2] for person in people]
        return (keypoints, scores), draw(image, people)

    return call_and_draw if callable(draw) else call


def compute_transform_params(image, input_size=512, multiple=64, factor=200):
    height, width = image.shape[:2]
    center = np.array([int(width / 2 + 0.5), int(height / 2 + 0.5)])
    size = compute_transform_size(image, input_size, multiple)
    scale = compute_transform_scale(image, size, factor)
    return center, scale, size


def compute_transform_size(image, input_size, multiple):
    short_side, long_side = np.sort(image.shape[:2])
    height = int(input_size)
    width = get_upper_multiple(input_size / short_side * long_side, multiple)
    size = np.array([width, height])
    if image.shape[1] < image.shape[0]:
        size[0], size[1] = size[1], size[0]
    return size


def get_upper_multiple(value, multiple):
    return int(np.ceil(value / multiple) * multiple)


def compute_transform_scale(image, size, factor):
    height, width = image.shape[:2]
    height_resized, width_resized = size
    if height < width:
        height_resized, width_resized = width_resized, height_resized
    height, width = np.sort([height, width])
    scale_height = height / factor
    scale_width = (width_resized / height_resized) * scale_height
    return np.array([scale_width, scale_height])


def build_affine(center, scale, size, inverse):
    source, destination = source_destination_points(center, scale, size)
    if inverse:
        source, destination = destination, source
    return solve_affine(source, destination)


def solve_affine(source, destination):
    source = np.asarray(source, np.float64)
    destination = np.asarray(destination, np.float64)
    homogeneous = np.concatenate([source, np.ones((3, 1))], axis=1)
    return np.linalg.solve(homogeneous, destination).T


def source_destination_points(center, scale, size, factor=200):
    if not isinstance(scale, (np.ndarray, list)):
        scale = np.array([scale, scale])
    return source_points(scale, center, factor), destination_points(size)


def source_points(scale, center, factor):
    center_width = (scale[0] * factor) / 2
    points = np.zeros((3, 2), dtype=np.float32)
    points[0] = center
    points[1] = center + np.array([0, -center_width], dtype=np.float32)
    points[2] = third_point(points[0], points[1])
    return points


def destination_points(size):
    center_width, center_height = np.array(size[:2]) / 2
    points = np.zeros((3, 2), dtype=np.float32)
    points[0] = [center_width, center_height]
    points[1] = [center_width, center_height - center_width]
    points[2] = third_point(points[0], points[1])
    return points


def third_point(point_a, point_b):
    difference = point_a - point_b
    return point_a + np.array([-difference[1], difference[0]], dtype=np.float32)


def transform_to_image(people, center, scale, shape):
    transform = build_affine(center, scale, shape, inverse=True)
    transformed = []
    for person in people:
        person = person.copy()
        for keypoint in person:
            point = np.array([keypoint[0], keypoint[1], 1.0])
            keypoint[0:2] = np.dot(transform, point)[:2]
        transformed.append(person[:, :3])
    return transformed


def EstimateHumanPose3D():
    model = paz.models.SimpleBaseline(weights="human36m")

    def call(keypoints2D):
        keypoints2D = np.array(keypoints2D)
        merged = merge_into_mean(keypoints2D, human36m.args_to_mean)
        filtered = filter_to_human36m(merged)
        standardized = standardize(filtered, human36m.data_mean2D,
                                   human36m.data_stdev2D)
        predicted = np.array(model(standardized))
        keypoints3D = destandardize_pose(predicted)
        return keypoints3D.reshape(-1, 32, 3)

    return call


def EstimateHumanPose(solver, camera_intrinsics, draw=None):
    estimate_keypoints2D = HigherHRNetHumanPose2D(draw=False)
    estimate_keypoints3D = EstimateHumanPose3D()
    if draw is None:
        draw = draw_human_pose6D

    def call(image):
        keypoints2D, scores = estimate_keypoints2D(image)
        if len(keypoints2D) == 0:
            return keypoints2D, [], []
        keypoints3D = estimate_keypoints3D(keypoints2D)
        poses3D, projection2D = paz.poses.optimize_human_pose3D(
            keypoints3D, np.array(keypoints2D), solver, camera_intrinsics)
        pose6D = paz.poses.human_pose3D_to_pose6D(poses3D[0])
        return keypoints2D, poses3D, pose6D

    def call_and_draw(image):
        keypoints2D, poses3D, pose6D = call(image)
        if len(keypoints2D) == 0:
            return (keypoints2D, poses3D, pose6D), image
        rotation, translation = pose6D
        image = draw(image, rotation, translation, camera_intrinsics)
        return (keypoints2D, poses3D, pose6D), image

    return call_and_draw if callable(draw) else call


def draw_human_pose6D(image, rotation, translation, camera_intrinsics,
                      length=60, offset=(50, 50)):
    image = np.array(image)
    points2D = paz.poses.project_to_image(
        rotation, np.array(translation), np.eye(3), camera_intrinsics)
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    offset = np.array(offset)
    for point, color in zip(points2D, colors):
        unit = (point / np.linalg.norm(point) * length).astype("int32")
        image = paz.draw.line(image, offset.tolist(),
                              (unit + offset).tolist(), color, 4)
    return image


def merge_into_mean(keypoints2D, args_to_mean):
    keypoints2D = keypoints2D.copy()
    for joint, (first, second) in args_to_mean.items():
        keypoints2D[:, joint] = (keypoints2D[:, first] + keypoints2D[:, second]) / 2  # fmt: skip
    return keypoints2D


def filter_to_human36m(keypoints2D):
    selected = keypoints2D[:, human36m.h36m_to_coco_joints2D, :]
    return selected.reshape(selected.shape[0], -1)


def standardize(data, mean, stdev):
    return (data - mean) / stdev


def destandardize_pose(predicted):
    data = predicted.reshape(-1, 48)
    rearranged = np.zeros((len(data), len(human36m.data_mean3D)))
    rearranged[:, human36m.dim_to_use3D] = data
    return rearranged * human36m.data_stdev3D + human36m.data_mean3D


def draw_human_skeletons(image, people, link_width=2, radius=3):
    image = np.array(image)
    for person in people:
        image = draw_skeleton(image, person, link_width, radius)
    return image


def draw_skeleton(image, person, link_width, radius):
    points = [(int(x), int(y)) for x, y in person[:, :2]]
    for link_arg, (parent, child) in enumerate(human_links):
        if person[parent, 2] > 0 and person[child, 2] > 0:
            color = link_colors[link_arg].tolist()
            image = paz.draw.line(image, points[parent], points[child],
                                  color, link_width)
    for keypoint_arg, point in enumerate(points):
        if person[keypoint_arg, 2] > 0:
            image = paz.draw.keypoint(image, point, [0, 255, 0], radius)
    return image
