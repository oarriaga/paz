import numpy as np
import paz
from paz.backend.poses import LEVENBERG_MARQUARDT
from paz.backend.poses import project_points3D
from paz.backend.poses import solve_PnP
from paz.backend.pinhole import build_cube_points3D


def draw_boxes3D(image, poses, points3D, camera, color, thickness=5, radius=2):
    for pose in poses:
        points2D = project_points3D(points3D, pose, camera)
        points2D = paz.to_numpy(points2D).astype(np.int32)
        paz.draw.cube(image, points2D, color, thickness, radius)
    return image


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
        cube = paz.to_numpy(build_cube_points3D(900, 1200, 800))
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
