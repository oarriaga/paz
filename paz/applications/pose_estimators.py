from collections import namedtuple
import cv2
import numpy as np
import jax.numpy as jp
import paz


UPNP = cv2.SOLVEPNP_UPNP
LEVENBERG_MARQUARDT = cv2.SOLVEPNP_ITERATIVE

Pose6D = namedtuple("Pose6D", ["rotation_vector", "translation"])


def build_cube_corners(width, height, depth):
    """Build the 3D points of a cube in the openCV coordinate system:
                               4--------1
                              /|       /|
                             / |      / |
                            3--------2  |
                            |  8_____|__5
                            | /      | /
                            |/       |/
                            7--------6

                   Z (depth)
                  /
                 /_____X (width)
                 |
                 |
                 Y (height)

    # Arguments
        height: float, height of the 3D box.
        width: float,  width of the 3D box.
        depth: float,  width of the 3D box.

    # Returns
        Numpy array of shape ``(8, 3)'' corresponding to 3D keypoints of a cube
    """
    half_height, half_width, half_depth = height / 2.0, width / 2.0, depth / 2.0
    point1 = [+half_width, -half_height, +half_depth]
    point2 = [+half_width, -half_height, -half_depth]
    point3 = [-half_width, -half_height, -half_depth]
    point4 = [-half_width, -half_height, +half_depth]
    point5 = [+half_width, +half_height, +half_depth]
    point6 = [+half_width, +half_height, -half_depth]
    point7 = [-half_width, +half_height, -half_depth]
    point8 = [-half_width, +half_height, +half_depth]
    points = [point1, point2, point3, point4, point5, point6, point7, point8]
    return jp.array(points)


def project_points3D(points3D, pose6D, camera):
    args = (pose6D.translation, camera.intrinsics, camera.distortion)
    points2D, _ = cv2.projectPoints(points3D, pose6D.rotation_vector, *args)
    points2D = jp.squeeze(points2D, axis=1)  # openCV shape (num_points, 1, 2)
    return points2D


def draw_boxes3D(image, poses, points3D, camera, color, thickness=5, radius=2):
    for pose in poses:
        points2D = project_points3D(points3D, pose, camera)
        points2D = paz.to_numpy(points2D).astype(np.int32)
        paz.draw.cube(image, points2D, color, thickness, radius)
    return image


def solve_PnP(points2D, points3D, camera, solver=LEVENBERG_MARQUARDT):
    points2D = np.array(points2D, np.float64).reshape((len(points3D), 1, 2))
    args = (camera.intrinsics, camera.distortion, None, None, False, solver)
    (_, rotation_vector, translation) = cv2.solvePnP(points3D, points2D, *args)
    return Pose6D(rotation_vector, translation)


def build_face_points3D():
    points3D = np.array(
        [
            [-220, 1138, 678],  # left--center-eye
            [+220, 1138, 678],  # right-center-eye
            [-131, 1107, 676],  # left--eye close to nose
            [-294, 1123, 610],  # left--eye close to ear
            [+131, 1107, 676],  # right-eye close to nose
            [+294, 1123, 610],  # right-eye close to ear
            [-106, 1224, 758],  # left--eyebrow close to nose
            [-375, 1208, 585],  # left--eyebrow close to ear
            [+106, 1224, 758],  # right-eyebrow close to nose
            [+375, 1208, 585],  # right-eyebrow close to ear
            [0.0, 919, 909],  # nose
            [-183, 683, 691],  # lefty-lip
            [+183, 683, 691],  # right-lip
            [0.0, 754, 826],  # up---lip
            [0.0, 645, 815],  # down-lip
        ]
    )

    return points3D - np.mean(points3D, axis=0)


def HeadPoseKeypointNet2D32(camera, box_scale=1.2, draw=None):
    detect = paz.models.HaarCascadeFrontalFaceDetector(draw=False)
    estimate_keypoints = paz.applications.FaceKeypointNet2D32(draw=False)
    points3D = build_face_points3D()
    camera.intrinsics = paz.to_numpy(camera.intrinsics)
    # camera.distortion = paz.to_numpy(camera.distortion)  # TODO
    solve_pose = paz.lock(solve_PnP, points3D, camera, LEVENBERG_MARQUARDT)

    if draw is None:
        cube = paz.to_numpy(build_cube_corners(900, 1200, 800))
        draw = paz.lock(draw_boxes3D, cube, camera, paz.draw.GREEN, 3, 5)

    def call(image):
        boxes = paz.detection.get_boxes(detect(image))
        boxes = paz.boxes.square(boxes)
        boxes = paz.boxes.scale(boxes, box_scale, box_scale)
        boxes = paz.cast(boxes, "int32")
        boxes = paz.boxes.remove_invalid(boxes)
        poses6D = []
        for box in boxes:
            keypoints = estimate_keypoints(paz.image.crop(image, box))
            keypoints = paz.points2D.shift_to_box_origin(keypoints, box)
            poses6D.append(solve_pose(keypoints))
        return poses6D

    return (lambda x: (y := call(x), draw(x, y))) if callable(draw) else call
