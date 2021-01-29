################################################
### The functions in this file are direct    ###
### copy of the ones present in the original ###
### Frustum PointNet work                    ###
################################################


import numpy as np
import tensorflow as tf
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# -----------------
# Global Constants
# -----------------

NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 8  # one cluster for each type
NUM_OBJECT_POINT = 512

g_type2class = {'Car': 0, 'Van': 1, 'Truck': 2, 'Pedestrian': 3,
                'Person_sitting': 4, 'Cyclist': 5, 'Tram': 6, 'Misc': 7}
g_class2type = {g_type2class[t]: t for t in g_type2class}
g_type2onehotclass = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
g_type_mean_size = {'Car': np.array([3.88311640418, 1.62856739989, 1.52563191462]),
                    'Van': np.array([5.06763659, 1.9007158, 2.20532825]),
                    'Truck': np.array([10.13586957, 2.58549199, 3.2520595]),
                    'Pedestrian': np.array([0.84422524, 0.66068622, 1.76255119]),
                    'Person_sitting': np.array([0.80057803, 0.5983815, 1.27450867]),
                    'Cyclist': np.array([1.76282397, 0.59706367, 1.73698127]),
                    'Tram': np.array([16.17150617, 2.53246914, 3.53079012]),
                    'Misc': np.array([3.64300781, 1.54298177, 1.92320313])}
g_mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3))  # size clustrs
for i in range(NUM_SIZE_CLUSTER):
    g_mean_size_arr[i, :] = g_type_mean_size[g_class2type[i]]


map_2d_detector = {
    1: 3,  # car
    2: 7,  # pedestrian
    3: 0,  # animal
    4: 6,  # other_vehicle
    5: 2,  # bus
    6: 5,  # motorcycle
    7: 8,  # truck
    8: 4,  # emergency_vehicle
    9: 1,  # bicycle
}


# -----------------
# TF Functions Helpers
# -----------------

def tf_gather_object_pc(point_cloud, mask, npoints=512):
    """ Gather object point clouds according to predicted masks.
    Input:
        point_cloud: TF tensor in shape (B,N,C)
        mask: TF tensor in shape (B,N) of 0 (not pick) or 1 (pick)
        npoints: int scalar, maximum number of points to keep (default: 512)
    Output:
        object_pc: TF tensor in shape (B,npoint,C)
        indices: TF int tensor in shape (B,npoint,2)
    """

    def mask_to_indices(mask):
        indices = np.zeros((mask.shape[0], npoints, 2), dtype=np.int32)
        for i in range(mask.shape[0]):
            pos_indices = np.where(mask[i, :] > 0.5)[0]
            # skip cases when pos_indices is empty
            if len(pos_indices) > 0:
                if len(pos_indices) > npoints:
                    choice = np.random.choice(len(pos_indices),
                                              npoints, replace=False)
                else:
                    choice = np.random.choice(len(pos_indices),
                                              npoints - len(pos_indices), replace=True)
                    choice = np.concatenate((np.arange(len(pos_indices)), choice))
                np.random.shuffle(choice)
                indices[i, :, 1] = pos_indices[choice]
            indices[i, :, 0] = i
        return indices

    indices = tf.py_function(mask_to_indices, [mask], tf.int32)
    object_pc = tf.gather_nd(point_cloud, indices)
    return object_pc, indices


def parse_data(raw_record):
    example = parse_frustum_point_record(raw_record)
    return example['frustum_point_cloud'], \
           tf.cast(example['one_hot_vec'], tf.float32), \
           tf.cast(example['seg_label'], tf.int32), \
           example['box3d_center'], \
           tf.cast(example['angle_class'], tf.int32), \
           example['angle_residual'], \
           tf.cast(example['size_class'], tf.int32), \
           example['size_residual']


def parse_frustum_point_record(tfexample_message: str):
    NUM_CLASS = 3
    NUM_POINT = 1024
    NUM_CHANNELS_OF_PC = 3

    keys_to_features = {
        "size_class": tf.io.FixedLenFeature((), tf.int64, tf.zeros((), tf.int64)),
        "size_residual": tf.io.FixedLenFeature((3,), tf.float32, tf.zeros((3,), tf.float32)),
        "seg_label": tf.io.FixedLenFeature((NUM_POINT,), tf.int64, tf.zeros((NUM_POINT,), tf.int64)),
        "frustum_point_cloud": tf.io.FixedLenFeature((NUM_POINT, NUM_CHANNELS_OF_PC), tf.float32),
        "rot_angle": tf.io.FixedLenFeature((), tf.float32, tf.zeros((), tf.float32)),
        "angle_class": tf.io.FixedLenFeature((), tf.int64, tf.zeros((), tf.int64)),
        "angle_residual": tf.io.FixedLenFeature((), tf.float32, tf.zeros((), tf.float32)),
        "one_hot_vec": tf.io.FixedLenFeature((NUM_CLASS,), tf.int64),
        "box3d_center": tf.io.FixedLenFeature((3,), tf.float32, tf.zeros((3,), tf.float32)),
    }
    parsed_example = tf.io.parse_single_example(tfexample_message, keys_to_features)
    return parsed_example


def get_box3d_corners_helper(centers, headings, sizes):
    """ TF layer. Input: (N,3), (N,), (N,3), Output: (N,8,3) """
    # print '-----', centers
    N = centers.get_shape()[0]
    l = tf.slice(sizes, [0, 0], [-1, 1])  # (N,1)
    w = tf.slice(sizes, [0, 1], [-1, 1])  # (N,1)
    h = tf.slice(sizes, [0, 2], [-1, 1])  # (N,1)
    # print l,w,h
    x_corners = tf.concat([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], axis=1)  # (N,8)
    y_corners = tf.concat([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2], axis=1)  # (N,8)
    z_corners = tf.concat([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], axis=1)  # (N,8)
    corners = tf.concat([tf.expand_dims(x_corners, 1), tf.expand_dims(y_corners, 1), tf.expand_dims(z_corners, 1)],
                        axis=1)  # (N,3,8)
    # print x_corners, y_corners, z_corners
    c = tf.cos(headings)
    s = tf.sin(headings)
    ones = tf.ones([N], dtype=tf.float32)
    zeros = tf.zeros([N], dtype=tf.float32)
    row1 = tf.stack([c, zeros, s], axis=1)  # (N,3)
    row2 = tf.stack([zeros, ones, zeros], axis=1)
    row3 = tf.stack([-s, zeros, c], axis=1)
    R = tf.concat([tf.expand_dims(row1, 1), tf.expand_dims(row2, 1), tf.expand_dims(row3, 1)], axis=1)  # (N,3,3)
    # print row1, row2, row3, R, N
    corners_3d = tf.matmul(R, corners)  # (N,3,8)
    corners_3d += tf.tile(tf.expand_dims(centers, 2), [1, 1, 8])  # (N,3,8)
    corners_3d = tf.transpose(corners_3d, perm=[0, 2, 1])  # (N,8,3)
    return corners_3d


def get_box3d_corners(center, heading_residuals, size_residuals):
    """ TF layer.
    Inputs:
        center: (B,3)
        heading_residuals: (B,NH)
        size_residuals: (B,NS,3)
    Outputs:
        box3d_corners: (B,NH,NS,8,3) tensor
    """
    batch_size = center.get_shape()[0]
    heading_bin_centers = tf.constant(np.arange(0, 2 * np.pi, 2 * np.pi / NUM_HEADING_BIN), dtype=tf.float32)  # (NH,)
    headings = heading_residuals + tf.expand_dims(heading_bin_centers, 0)  # (B,NH)

    mean_sizes = tf.expand_dims(tf.constant(g_mean_size_arr, dtype=tf.float32), 0) + size_residuals  # (B,NS,1)
    sizes = mean_sizes + size_residuals  # (B,NS,3)
    sizes = tf.tile(tf.expand_dims(sizes, 1), [1, NUM_HEADING_BIN, 1, 1])  # (B,NH,NS,3)
    headings = tf.tile(tf.expand_dims(headings, -1), [1, 1, NUM_SIZE_CLUSTER])  # (B,NH,NS)
    centers = tf.tile(tf.expand_dims(tf.expand_dims(center, 1), 1),
                      [1, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 1])  # (B,NH,NS,3)

    N = batch_size * NUM_HEADING_BIN * NUM_SIZE_CLUSTER
    corners_3d = get_box3d_corners_helper(tf.reshape(centers, [N, 3]), tf.reshape(headings, [N]),
                                          tf.reshape(sizes, [N, 3]))

    return tf.reshape(corners_3d, [batch_size, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 8, 3])


def huber_loss(error, delta):
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic ** 2 + delta * linear
    return tf.reduce_mean(losses)


def parse_output_to_tensors(output, end_points):
    """ Parse batch output to separate tensors (added to end_points)
    Input:
        output: TF tensor in shape (B,3+2*NUM_HEADING_BIN+4*NUM_SIZE_CLUSTER)
        end_points: dict
    Output:
        end_points: dict
    """
    batch_size = output.get_shape()[0]
    center = tf.slice(output, [0, 0], [-1, 3])
    end_points['center_boxnet'] = center

    heading_scores = tf.slice(output, [0, 3], [-1, NUM_HEADING_BIN])
    heading_residuals_normalized = tf.slice(output, [0, 3 + NUM_HEADING_BIN],
                                            [-1, NUM_HEADING_BIN])
    end_points['heading_scores'] = heading_scores  # BxNUM_HEADING_BIN
    end_points['heading_residuals_normalized'] = \
        heading_residuals_normalized  # BxNUM_HEADING_BIN (-1 to 1)
    end_points['heading_residuals'] = \
        heading_residuals_normalized * (np.pi / NUM_HEADING_BIN)  # BxNUM_HEADING_BIN

    size_scores = tf.slice(output, [0, 3 + NUM_HEADING_BIN * 2],
                           [-1, NUM_SIZE_CLUSTER])  # BxNUM_SIZE_CLUSTER
    size_residuals_normalized = tf.slice(output,
                                         [0, 3 + NUM_HEADING_BIN * 2 + NUM_SIZE_CLUSTER], [-1, NUM_SIZE_CLUSTER * 3])
    size_residuals_normalized = tf.reshape(size_residuals_normalized,
                                           [batch_size, NUM_SIZE_CLUSTER, 3])  # BxNUM_SIZE_CLUSTERx3
    end_points['size_scores'] = size_scores
    end_points['size_residuals_normalized'] = size_residuals_normalized
    end_points['size_residuals'] = size_residuals_normalized * \
                                   tf.expand_dims(tf.constant(g_mean_size_arr, dtype=tf.float32), 0)

    return end_points


def point_cloud_masking(point_cloud, logits, end_points, xyz_only=True):
    """ Select point cloud with predicted 3D mask,
    translate coordinates to the masked points centroid.

    Used from Frustum PointNet

    Input:
        point_cloud: TF tensor in shape (B,N,C)
        logits: TF tensor in shape (B,N,2)
        end_points: dict
        xyz_only: boolean, if True only return XYZ channels
    Output:
        object_point_cloud: TF tensor in shape (B,M,3)
            for simplicity we only keep XYZ here
            M = NUM_OBJECT_POINT as a hyper-parameter
        mask_xyz_mean: TF tensor in shape (B,3)
    """
    batch_size = point_cloud.get_shape()[0]
    num_point = point_cloud.get_shape()[1]
    mask = tf.slice(logits, [0, 0, 0], [-1, -1, 1]) < \
           tf.slice(logits, [0, 0, 1], [-1, -1, 1])
    mask = tf.cast(mask, dtype=tf.float32)  # BxNx1
    mask_count = tf.tile(tf.math.reduce_sum(mask, axis=1, keepdims=True), [1, 1, 3])  # Bx1x3
    point_cloud_xyz = tf.slice(point_cloud, [0, 0, 0], [-1, -1, 3])  # BxNx3
    mask_xyz_mean = tf.math.reduce_sum(tf.tile(mask, [1, 1, 3]) * point_cloud_xyz, axis=1, keepdims=True)  # Bx1x3
    mask = tf.squeeze(mask, axis=[2])  # BxN
    end_points['mask'] = mask
    mask_xyz_mean = mask_xyz_mean / tf.maximum(mask_count, 1)  # Bx1x3

    # Translate to masked points' centroid
    point_cloud_xyz_stage1 = point_cloud_xyz - \
                             tf.tile(mask_xyz_mean, [1, num_point, 1])

    if xyz_only:
        point_cloud_stage1 = point_cloud_xyz_stage1
    else:
        point_cloud_features = tf.slice(point_cloud, [0, 0, 3], [-1, -1, -1])
        point_cloud_stage1 = tf.concat([point_cloud_xyz_stage1, point_cloud_features], axis=-1)
    num_channels = point_cloud_stage1.get_shape()[2]

    object_point_cloud, _ = tf_gather_object_pc(point_cloud_stage1,
                                                mask, NUM_OBJECT_POINT)
    object_point_cloud.set_shape([batch_size, NUM_OBJECT_POINT, num_channels])

    return object_point_cloud, tf.squeeze(mask_xyz_mean, axis=1), end_points

