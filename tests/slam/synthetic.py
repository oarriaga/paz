import functools
from collections import namedtuple

import jax
import jax.numpy as jp
from jax.experimental import enable_x64

import paz


def in_double_precision(function):
    # scenes need float64 geometry, but flipping the global x64 flag
    # would leak into every other test collected in the same process
    @functools.wraps(function)
    def wrapped(*args):
        with enable_x64():
            return function(*args)

    return wrapped

TwoViewScene = namedtuple(
    "TwoViewScene",
    ["intrinsics_A", "intrinsics_B", "pose_A", "pose_B", "points3D",
     "points_A", "points_B", "valid_mask", "inlier_mask"],
)

PnPScene = namedtuple(
    "PnPScene",
    ["intrinsics", "pose", "points3D", "points2D", "valid_mask",
     "inlier_mask"],
)

BundleScene = namedtuple(
    "BundleScene",
    ["intrinsics", "poses", "noisy_poses", "points3D", "noisy_points3D",
     "observations", "visibility"],
)

StereoSequence = namedtuple(
    "StereoSequence",
    ["intrinsics", "left_to_right", "poses", "points3D",
     "observations_left", "observations_right", "visibility_left",
     "visibility_right", "outlier_mask_left", "outlier_mask_right",
     "timestamps"],
)

DegenerateScenes = namedtuple(
    "DegenerateScenes",
    ["planar", "pure_rotation", "low_parallax", "behind_camera",
     "collinear", "insufficient"],
)


@in_double_precision
def build_two_view_scene(key, num_points, noise_stdv, outlier_fraction):
    point_key, noise_A, noise_B, outlier_key = jax.random.split(key, 4)
    box = ((-2.0, -1.5, 4.0), (2.0, 1.5, 8.0))
    points3D = sample_box_points(point_key, num_points, *box)
    pose_A, pose_B = build_standard_pair()
    intrinsics_A = build_intrinsics(500.0, 500.0, 320.0, 240.0)
    intrinsics_B = build_intrinsics(540.0, 530.0, 320.0, 240.0)
    points_A = project_points(intrinsics_A, pose_A, points3D)
    points_B = project_points(intrinsics_B, pose_B, points3D)
    points_A = add_pixel_noise(noise_A, points_A, noise_stdv)
    points_B = add_pixel_noise(noise_B, points_B, noise_stdv)
    corrupt = (outlier_key, intrinsics_B, points_B, outlier_fraction)
    points_B, inlier_mask = corrupt_with_outliers(*corrupt)
    valid_mask = jp.ones(num_points, dtype=bool)
    scene = (intrinsics_A, intrinsics_B, pose_A, pose_B, points3D,
             points_A, points_B, valid_mask, inlier_mask)
    return TwoViewScene(*scene)


@in_double_precision
def build_pnp_scene(key, num_points, noise_stdv, outlier_fraction):
    point_key, noise_key, outlier_key = jax.random.split(key, 3)
    box = ((-2.0, -1.5, 4.0), (2.0, 1.5, 8.0))
    points3D = sample_box_points(point_key, num_points, *box)
    target = jp.array([0.0, 0.0, 6.0])
    pose = look_at_pose(jp.array([0.6, -0.4, 0.3]), target)
    intrinsics = build_intrinsics(500.0, 500.0, 320.0, 240.0)
    points2D = project_points(intrinsics, pose, points3D)
    points2D = add_pixel_noise(noise_key, points2D, noise_stdv)
    corrupt = (outlier_key, intrinsics, points2D, outlier_fraction)
    points2D, inlier_mask = corrupt_with_outliers(*corrupt)
    valid_mask = jp.ones(num_points, dtype=bool)
    scene = (intrinsics, pose, points3D, points2D, valid_mask,
             inlier_mask)
    return PnPScene(*scene)


@in_double_precision
def build_bundle_adjustment_scene(
    key, num_poses, num_points, noise_stdv, pose_noise, point_noise
):
    keys = jax.random.split(key, 5)
    intrinsics = build_intrinsics(500.0, 500.0, 320.0, 240.0)
    box = ((-3.0, -2.0, -2.0), (3.0, 2.0, 2.0))
    points3D = sample_box_points(keys[0], num_points, *box)
    poses = build_arc_poses(num_poses, 9.0, 0.4)
    observe = (keys[1], intrinsics, poses, points3D, noise_stdv)
    observations, in_view = observe_points(*observe)
    dropout = jax.random.bernoulli(keys[2], 0.15, in_view.shape)
    visibility = in_view & ~dropout
    noisy_poses = perturb_poses(keys[3], poses, pose_noise)
    noise3D = point_noise * jax.random.normal(keys[4], points3D.shape)
    noisy_points3D = points3D + noise3D
    scene = (intrinsics, poses, noisy_poses, points3D, noisy_points3D,
             observations, visibility)
    return BundleScene(*scene)


@in_double_precision
def build_stereo_sequence(
    key, num_frames, num_points, noise_stdv, outlier_fraction
):
    point_key, left_key, right_key = jax.random.split(key, 3)
    intrinsics = build_intrinsics(500.0, 500.0, 320.0, 240.0)
    left_to_right = paz.SE3.translation(jp.array([-0.15, 0.0, 0.0]))
    box = ((-3.0, -2.0, -2.0), (3.0, 2.0, 2.0))
    points3D = sample_box_points(point_key, num_points, *box)
    poses = build_arc_poses(num_frames, 9.0, 0.5)
    poses_right = jp.stack([left_to_right @ pose for pose in poses])
    args_left = (left_key, intrinsics, poses, points3D, noise_stdv,
                 outlier_fraction, 0.15)
    args_right = (right_key, intrinsics, poses_right, points3D,
                  noise_stdv, outlier_fraction, 0.15)
    left = observe_camera(*args_left)
    right = observe_camera(*args_right)
    observations_left, visibility_left, outliers_left = left
    observations_right, visibility_right, outliers_right = right
    timestamps = jp.arange(num_frames) / 30.0
    sequence = (intrinsics, left_to_right, poses, points3D,
                observations_left, observations_right, visibility_left,
                visibility_right, outliers_left, outliers_right,
                timestamps)
    return StereoSequence(*sequence)


@in_double_precision
def build_degenerate_scenes(key):
    keys = jax.random.split(key, 6)
    scenes = (build_planar_scene(keys[0]),
              build_pure_rotation_scene(keys[1]),
              build_low_parallax_scene(keys[2]),
              build_behind_camera_scene(keys[3]),
              build_collinear_scene(keys[4]),
              build_insufficient_scene(keys[5]))
    return DegenerateScenes(*scenes)


def build_planar_scene(key):
    box = ((-2.0, -1.5, 6.0), (2.0, 1.5, 6.0001))
    points3D = sample_box_points(key, 60, *box)
    pose_A, pose_B = build_standard_pair()
    return assemble_two_view(points3D, pose_A, pose_B, full_mask(60))


def build_pure_rotation_scene(key):
    box = ((-2.0, -1.5, 4.0), (2.0, 1.5, 8.0))
    points3D = sample_box_points(key, 60, *box)
    origin = jp.array([0.0, 0.0, 0.0])
    pose_A = look_at_pose(origin, jp.array([0.0, 0.0, 6.0]))
    pose_B = look_at_pose(origin, jp.array([1.5, 0.3, 6.0]))
    return assemble_two_view(points3D, pose_A, pose_B, full_mask(60))


def build_low_parallax_scene(key):
    box = ((-2.0, -1.5, 4.0), (2.0, 1.5, 8.0))
    points3D = sample_box_points(key, 60, *box)
    target = jp.array([0.0, 0.0, 6.0])
    pose_A = look_at_pose(jp.array([0.0, 0.0, 0.0]), target)
    pose_B = look_at_pose(jp.array([1e-4, 0.0, 0.0]), target)
    return assemble_two_view(points3D, pose_A, pose_B, full_mask(60))


def build_behind_camera_scene(key):
    box = ((-2.0, -1.5, 4.0), (2.0, 1.5, 8.0))
    points3D = sample_box_points(key, 60, *box)
    pose_A = look_at_pose(jp.zeros(3), jp.array([0.0, 0.0, 6.0]))
    pose_B = look_at_pose(jp.array([0.0, 0.0, 6.0]),
                          jp.array([0.0, 0.0, 12.0]))
    return assemble_two_view(points3D, pose_A, pose_B, full_mask(60))


def build_collinear_scene(key):
    line_parameters = jax.random.uniform(key, (60, 1))
    base = jp.array([-1.5, -1.0, 4.5])
    direction = jp.array([3.0, 2.0, 3.0])
    points3D = base + line_parameters * direction
    pose_A, pose_B = build_standard_pair()
    return assemble_two_view(points3D, pose_A, pose_B, full_mask(60))


def build_insufficient_scene(key):
    box = ((-2.0, -1.5, 4.0), (2.0, 1.5, 8.0))
    points3D = sample_box_points(key, 12, *box)
    pose_A, pose_B = build_standard_pair()
    valid_mask = jp.arange(12) < 6
    return assemble_two_view(points3D, pose_A, pose_B, valid_mask)


def assemble_two_view(points3D, pose_A, pose_B, valid_mask):
    intrinsics_A = build_intrinsics(500.0, 500.0, 320.0, 240.0)
    intrinsics_B = build_intrinsics(500.0, 500.0, 320.0, 240.0)
    points_A = project_points(intrinsics_A, pose_A, points3D)
    points_B = project_points(intrinsics_B, pose_B, points3D)
    in_front_A = compute_depths(pose_A, points3D) > 0.0
    in_front_B = compute_depths(pose_B, points3D) > 0.0
    inlier_mask = valid_mask & in_front_A & in_front_B
    scene = (intrinsics_A, intrinsics_B, pose_A, pose_B, points3D,
             points_A, points_B, valid_mask, inlier_mask)
    return TwoViewScene(*scene)


@in_double_precision
def compute_relative_transform(pose_A, pose_B):
    return pose_B @ paz.SE3.invert(pose_A)


@in_double_precision
def compute_essential(pose_A, pose_B):
    relative = compute_relative_transform(pose_A, pose_B)
    rotation = paz.SE3.get_rotation_matrix(relative)
    translation = paz.SE3.get_position_vector(relative)
    return paz.SO3.hat(translation) @ rotation


@in_double_precision
def compute_fundamental(intrinsics_A, intrinsics_B, pose_A, pose_B):
    essential = compute_essential(pose_A, pose_B)
    inverse_A = jp.linalg.inv(intrinsics_A)
    inverse_B = jp.linalg.inv(intrinsics_B)
    return inverse_B.T @ essential @ inverse_A


def build_standard_pair():
    target = jp.array([0.0, 0.0, 6.0])
    pose_A = look_at_pose(jp.array([0.0, 0.0, 0.0]), target)
    pose_B = look_at_pose(jp.array([1.2, 0.5, 0.8]), target)
    return pose_A, pose_B


def build_arc_poses(num_poses, radius, max_angle):
    angles = jp.linspace(-max_angle, max_angle, num_poses)
    poses = []
    for angle in angles:
        x = radius * jp.sin(angle)
        y = 0.4 * jp.sin(2.0 * angle)
        z = -radius * jp.cos(angle)
        poses.append(look_at_pose(jp.array([x, y, z]), jp.zeros(3)))
    return jp.stack(poses)


def look_at_pose(camera_origin, target_origin):
    # flip the graphics look-at (-z forward) to positive-depth vision
    world_up = jp.array([0.0, 1.0, 0.0])
    view = paz.SE3.view_transform(camera_origin, target_origin, world_up)
    rotation = nearest_rotation(view[:3, :3])
    translation = -rotation @ camera_origin
    pose = paz.SE3.to_affine_matrix(rotation, translation)
    return paz.SE3.rotation_x(jp.pi) @ pose


def nearest_rotation(matrix):
    # view_transform rows lose unit norm when the forward axis is not
    # orthogonal to world up; snap to the closest true rotation
    U, _, Vt = jp.linalg.svd(matrix)
    return U @ Vt


def observe_camera(
    key, intrinsics, poses, points3D, noise_stdv, outlier_fraction,
    dropout_rate
):
    keys = jax.random.split(key, 4)
    observe = (keys[0], intrinsics, poses, points3D, noise_stdv)
    observations, in_view = observe_points(*observe)
    dropout = jax.random.bernoulli(keys[1], dropout_rate, in_view.shape)
    visibility = in_view & ~dropout
    outliers = jax.random.bernoulli(keys[2], outlier_fraction,
                                    in_view.shape)
    outlier_mask = outliers & visibility
    H, W = paz.pinhole.get_image_size(intrinsics)
    pixels = jax.random.uniform(keys[3], observations.shape)
    pixels = pixels * jp.array([float(W), float(H)])
    observations = jp.where(outlier_mask[..., None], pixels, observations)
    return observations, visibility, outlier_mask


def observe_points(key, intrinsics, poses, points3D, noise_stdv):
    observations, in_view = [], []
    for pose in poses:
        projected = project_points(intrinsics, pose, points3D)
        visible = compute_in_view(intrinsics, pose, points3D, projected)
        observations.append(projected)
        in_view.append(visible)
    observations = jp.stack(observations)
    noise = noise_stdv * jax.random.normal(key, observations.shape)
    return observations + noise, jp.stack(in_view)


def compute_in_view(intrinsics, pose, points3D, points2D):
    H, W = paz.pinhole.get_image_size(intrinsics)
    depths = compute_depths(pose, points3D)
    u, v = points2D[:, 0], points2D[:, 1]
    in_image = (u >= 0.0) & (u < W) & (v >= 0.0) & (v < H)
    return in_image & (depths > 0.1)


def perturb_poses(key, poses, pose_noise):
    tangents = pose_noise * jax.random.normal(key, (len(poses), 6))
    noisy = []
    for pose, tangent in zip(poses, tangents):
        noisy.append(paz.SE3.exp(paz.SE3.hat(tangent)) @ pose)
    return jp.stack(noisy)


def corrupt_with_outliers(key, intrinsics, points2D, outlier_fraction):
    flag_key, pixel_key = jax.random.split(key)
    num_points = len(points2D)
    flags = sample_outlier_flags(flag_key, num_points, outlier_fraction)
    pixels = sample_image_points(pixel_key, intrinsics, num_points)
    corrupted = jp.where(flags[:, None], pixels, points2D)
    return corrupted, ~flags


def sample_outlier_flags(key, num_points, outlier_fraction):
    num_outliers = int(round(outlier_fraction * num_points))
    order = jax.random.permutation(key, num_points)
    flags = jp.zeros(num_points, dtype=bool)
    return flags.at[order[:num_outliers]].set(True)


def sample_image_points(key, intrinsics, num_points):
    H, W = paz.pinhole.get_image_size(intrinsics)
    samples = jax.random.uniform(key, (num_points, 2))
    return samples * jp.array([float(W), float(H)])


def sample_box_points(key, num_points, min_corner, max_corner):
    low = jp.array(min_corner)
    high = jp.array(max_corner)
    samples = jax.random.uniform(key, (num_points, 3))
    return low + samples * (high - low)


def add_pixel_noise(key, points2D, noise_stdv):
    return points2D + noise_stdv * jax.random.normal(key, points2D.shape)


def project_points(intrinsics, pose, points3D):
    camera_matrix = paz.pinhole.make_camera_matrix(intrinsics, pose)
    project = jax.vmap(paz.pinhole.project_to_2D, in_axes=(None, 0))
    return project(camera_matrix, points3D)


def compute_depths(pose, points3D):
    points_camera = paz.algebra.transform_points(pose, points3D)
    return points_camera[:, 2]


def build_intrinsics(focal_x, focal_y, center_x, center_y):
    return jp.array([
        [focal_x, 0.0, center_x],
        [0.0, focal_y, center_y],
        [0.0, 0.0, 1.0],
    ])


def full_mask(num_points):
    return jp.ones(num_points, dtype=bool)
