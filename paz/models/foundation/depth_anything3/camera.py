"""Camera decoder: camera token to world-to-camera extrinsics and intrinsics.

Quaternions are scalar-last (xyzw). The decoder predicts a camera-to-world
pose encoding; extrinsics are returned as the world-to-camera inverse.
"""
from keras import ops
from keras.layers import Dense, ReLU

from paz.models.transformers.attention import kernel

FOCAL_EPSILON = 1e-6


def build_camera_decoder(camera_token, hidden_size, image_shape):
    pose_encoding = decode_pose_encoding(camera_token, hidden_size)
    return pose_encoding_to_camera(pose_encoding, image_shape)


def decode_pose_encoding(camera_token, hidden_size):
    hidden = dense(camera_token, hidden_size, "cam_dec_fc1")
    hidden = ReLU()(hidden)
    hidden = dense(hidden, hidden_size, "cam_dec_fc2")
    hidden = ReLU()(hidden)
    translation = dense(hidden, 3, "cam_dec_t")
    quaternion = dense(hidden, 4, "cam_dec_qvec")
    field_of_view = ReLU()(dense(hidden, 2, "cam_dec_fov"))
    return ops.concatenate([translation, quaternion, field_of_view], axis=-1)


def pose_encoding_to_camera(pose_encoding, image_shape):
    translation = pose_encoding[..., :3]
    quaternion = pose_encoding[..., 3:7]
    field_of_view = pose_encoding[..., 7:]
    rotation = quaternion_to_matrix(quaternion)
    camera_to_world = ops.concatenate([rotation, translation[..., None]],
                                      axis=-1)
    extrinsics = invert_transform(camera_to_world)
    intrinsics = intrinsics_from_field_of_view(field_of_view, image_shape)
    return extrinsics, intrinsics


def quaternion_to_matrix(quaternion):
    i, j, k, r = ops.unstack(quaternion, axis=-1)
    two_s = 2.0 / ops.sum(quaternion * quaternion, axis=-1)
    row0 = ops.stack([1 - two_s * (j * j + k * k), two_s * (i * j - k * r),
                      two_s * (i * k + j * r)], axis=-1)
    row1 = ops.stack([two_s * (i * j + k * r), 1 - two_s * (i * i + k * k),
                      two_s * (j * k - i * r)], axis=-1)
    row2 = ops.stack([two_s * (i * k - j * r), two_s * (j * k + i * r),
                      1 - two_s * (i * i + j * j)], axis=-1)
    return ops.stack([row0, row1, row2], axis=-2)


def invert_transform(camera_to_world):
    rotation = camera_to_world[..., :3, :3]
    translation = camera_to_world[..., :3, 3:]
    transposed = ops.swapaxes(rotation, -1, -2)
    inverse_translation = -ops.matmul(transposed, translation)
    return ops.concatenate([transposed, inverse_translation], axis=-1)


def intrinsics_from_field_of_view(field_of_view, image_shape):
    height, width = image_shape[0], image_shape[1]
    focal_y = (height / 2.0) / clamp_tan(field_of_view[..., 0])
    focal_x = (width / 2.0) / clamp_tan(field_of_view[..., 1])
    return stack_intrinsics(focal_x, focal_y, width / 2.0, height / 2.0)


def stack_intrinsics(focal_x, focal_y, center_x, center_y):
    zero = ops.zeros_like(focal_x)
    one = ops.ones_like(focal_x)
    row0 = ops.stack([focal_x, zero, zero + center_x], axis=-1)
    row1 = ops.stack([zero, focal_y, zero + center_y], axis=-1)
    row2 = ops.stack([zero, zero, one], axis=-1)
    return ops.stack([row0, row1, row2], axis=-2)


def clamp_tan(angle):
    return ops.maximum(ops.tan(angle / 2.0), FOCAL_EPSILON)


def dense(tokens, units, name):
    return Dense(units, use_bias=True, kernel_initializer=kernel(),
                 name=name)(tokens)
