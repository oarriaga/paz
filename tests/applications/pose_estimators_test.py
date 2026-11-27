import jax.numpy as jp
import numpy as np

import paz
from paz.applications.pose_estimators import HeadPoseKeypointNet2D32
from paz.applications.pose_estimators import build_face_points3D


def build_fake_face_detector(draw):
    """Detects one face next to the left border, padded like Haar cascades."""
    detections = jp.array([[8, 21, 142, 155, 0], [-1, -1, -1, -1, 0]])
    return lambda image: detections


def build_fake_keypoint_estimator(draw):
    keypoints = build_face_points3D()[:, :2] / 20.0 + 80.0
    return lambda image: keypoints


def test_HeadPoseKeypointNet2D32_keeps_face_at_image_border(monkeypatch):
    name = "HaarCascadeFrontalFaceDetector"
    monkeypatch.setattr(paz.models, name, build_fake_face_detector)
    name = "FaceKeypointNet2D32"
    monkeypatch.setattr(paz.applications, name, build_fake_keypoint_estimator)
    camera = paz.backend.camera.Camera()
    camera.intrinsics = np.array([[200.0, 0, 100], [0, 200.0, 90], [0, 0, 1]])
    estimate_poses = HeadPoseKeypointNet2D32(camera, draw=False)
    poses6D = estimate_poses(np.zeros((180, 200, 3), "uint8"))
    assert len(poses6D) == 1
