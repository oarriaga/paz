from collections import namedtuple

import jax
import jax.numpy as jp

from paz.backend import pinhole
from paz.backend.lie import SE3

PROBLEM_FIELDS = ("poses", "pose_active", "landmarks", "landmark_active",
                  "intrinsics", "rig_extrinsics", "observations_uv",
                  "observation_pose", "observation_landmark",
                  "observation_camera", "observation_weight",
                  "observation_active")
BundleProblem = namedtuple("BundleProblem", PROBLEM_FIELDS)


def compute_observation_residuals(problem):
    def residual(uv, pose_index, landmark_index, camera_index, active):
        pose = problem.poses[pose_index]
        landmark = problem.landmarks[landmark_index]
        intrinsics = problem.intrinsics[camera_index]
        camera_pose = problem.rig_extrinsics[camera_index] @ pose
        projected = project_point(intrinsics, camera_pose, landmark)
        return active * (projected - uv)

    return jax.vmap(residual)(*unpack_observations(problem))


def compute_observation_jacobians(problem):
    def blocks(uv, pose_index, landmark_index, camera_index, active):
        pose = problem.poses[pose_index]
        landmark = problem.landmarks[landmark_index]
        intrinsics = problem.intrinsics[camera_index]
        extrinsic = problem.rig_extrinsics[camera_index]

        def residual(delta_pose, delta_landmark):
            camera_pose = extrinsic @ SE3.retract(pose, delta_pose)
            point = landmark + delta_landmark
            return project_point(intrinsics, camera_pose, point) - uv

        zeros = (jp.zeros(6, pose.dtype), jp.zeros(3, landmark.dtype))
        pose_block, landmark_block = jax.jacfwd(residual, (0, 1))(*zeros)
        return active * pose_block, active * landmark_block

    return jax.vmap(blocks)(*unpack_observations(problem))


def unpack_observations(problem):
    active = problem.observation_active.astype(problem.landmarks.dtype)
    return (problem.observations_uv, problem.observation_pose,
            problem.observation_landmark, problem.observation_camera,
            active)


def project_point(intrinsics, camera_pose, point3D):
    camera_matrix = pinhole.make_camera_matrix(intrinsics, camera_pose)
    pixel = camera_matrix @ jp.append(point3D, 1.0)
    depth = pixel[2]
    safe_depth = jp.where(jp.abs(depth) < 1e-8, 1e-8, depth)
    return pixel[:2] / safe_depth
