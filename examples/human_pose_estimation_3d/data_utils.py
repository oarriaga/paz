"""Utility functions for dealing with human3.6m data."""
import os
import numpy as np
import json
from backend import destandardize, filter_keypoints

# Joints in H3.6M -- data has 32 joints, but only 17  that move
H36M_NAMES = [''] * 32
H36M_NAMES[0] = 'Hip'
H36M_NAMES[1] = 'RHip'
H36M_NAMES[2] = 'RKnee'
H36M_NAMES[3] = 'RFoot'
H36M_NAMES[6] = 'LHip'
H36M_NAMES[7] = 'LKnee'
H36M_NAMES[8] = 'LFoot'
H36M_NAMES[12] = 'Spine'
H36M_NAMES[13] = 'Thorax'
H36M_NAMES[14] = 'Neck/Nose'
H36M_NAMES[15] = 'Head'
H36M_NAMES[17] = 'LShoulder'
H36M_NAMES[18] = 'LElbow'
H36M_NAMES[19] = 'LWrist'
H36M_NAMES[25] = 'RShoulder'
H36M_NAMES[26] = 'RElbow'
H36M_NAMES[27] = 'RWrist'

# Joints in COCO, 2D poses from HigherHRNet --> data has 17 joints; 
# these are the indices. Hip,12,14,16,11,13,15,
# Spine,Thorax,0,Head,5,7,9,6,8,10

COCO_NAMES = [''] * 17
COCO_NAMES[0] = 'Head'  # Nose renamed as head
COCO_NAMES[1] = 'Thorax'
COCO_NAMES[2] = 'Spine'
COCO_NAMES[4] = 'Hip'
COCO_NAMES[5] = 'LShoulder'
COCO_NAMES[6] = 'RShoulder'
COCO_NAMES[7] = 'LElbow'
COCO_NAMES[8] = 'RElbow'
COCO_NAMES[9] = 'LWrist'
COCO_NAMES[10] = 'RWrist'
COCO_NAMES[11] = 'LHip'
COCO_NAMES[12] = 'RHip'
COCO_NAMES[13] = 'LKnee'
COCO_NAMES[14] = 'RKnee'
COCO_NAMES[15] = 'LFoot'
COCO_NAMES[16] = 'RFoot'
valid_joints = [0, 1, 2, 3, 6, 7, 8, 12, 13, 15, 17, 18, 19, 25, 26, 27]


def poses2D_permutation(poses2D):
    poses2D = np.reshape(poses2D, (poses2D.shape[0], -1))
    poses2D[:, 2:4] = (poses2D[:, 10:12] + poses2D[:, 12:14]) / 2
    poses2D[:, 8:10] = (poses2D[:, 22:24] + poses2D[:, 24:26]) / 2
    poses2D[:, 4:6] = (poses2D[:, 2:4] + poses2D[:, 8:10]) / 2
    poses2D = np.reshape(poses2D, [poses2D.shape[0], 17, 2])
    return poses2D


def read_json_file(filename):
    """
    reads from a json file and saves the result in a list named data
    """
    with open(filename, 'r') as fp:
        data = json.load(fp)
    return np.array(data)


def filter_joints_3D(poses3D, joints=valid_joints):
    """Selects 16 moving joints (Neck/Nose excluded) from 32 predicted
    joints in 3D

    # Arguments
        poses3D: Nx96 points in camera coordinates        
    # Returns
        moving_joints_3D: Nx48 points (moving joints)
    """
    N = len(poses3D)
    poses3D = np.reshape(poses3D, [N, 32, 3])
    filtered_joints_3D = filter_keypoints(poses3D, valid_joints)
    filtered_joints_3D = filtered_joints_3D.reshape(len(poses3D), -1)
    return filtered_joints_3D


def load_joints_2D(joints2D, joint_names=H36M_NAMES):
    poses2D = np.array(joints2D)
    COCO_TO_GT_PERM = []
    poses2D = poses2D_permutation(poses2D)
    for name in joint_names:
        if name != '' and name in COCO_NAMES:
            COCO_TO_GT_PERM.append(COCO_NAMES.index(name))
    COCO_TO_GT_PERM = np.array(COCO_TO_GT_PERM)
    poses = poses2D[:, COCO_TO_GT_PERM, :]
    poses = np.reshape(poses, [poses.shape[0], -1])
    return poses


def load_params():
    """Loads normalization statistics: mean and stdev, dimensions 
    used and ignored 

    # Returns
        data_mean: nxd np array with the mean of the data
        data_std: nxd np array with the standard deviation of the data
        dim_to_use: nxd np array of dimensions used in the model
        dim_to_ignore: nxd np array of dimensions not used in the model
    """
    root = os.path.dirname(os.path.abspath(__file__))
    data_mean2D = read_json_file(
        os.path.join(root, 'mean_2D.json'))
    data_std2D = read_json_file(os.path.join(root, 'std_2D.json'))
    data_mean3D = read_json_file(
        os.path.join(root, 'data_mean_3d.json'))
    data_std3D = read_json_file(os.path.join(root, 'data_std_3d.json'))
    dim_to_use3D = read_json_file(
        os.path.join(root, 'dim_to_use_3d.json'))
    return data_mean2D, data_std2D, \
        data_mean3D, data_std3D, dim_to_use3D


def unnormalize_data(normalized_data, mean, stdd,
                     valid):
    """Un-normalizes a matrix whose mean has been substracted and 
       that has been divided by standard deviation. Some dimensions 
       might also be missing

    # Arguments
        normalized_data: nxd matrix to unnormalize
        mean: array with the mean of the data
        std: array with the standard deviation of the data
        valid: list of dimensions to keep in the data        
    # Returns
        unnormalized_data: the input normalized_data, but unnormalized
    """
    data = get_data(normalized_data, mean, valid)
    unnormalized_data = destandardize(data, mean, stdd)
    return unnormalized_data


def get_data(normalized_data, mean, valid):
    """parse data

    # Arguments
        normalized_data: nxd matrix to unnormalize
        mean:  nxd np array with the mean of the data
        valid: array of dimensions to be used     
    # Returns
        data: data to e unormalized
    """
    length = len(normalized_data)
    columns = len(mean)
    data = np.zeros((length, columns), dtype=np.float32)
    data[:, valid] = normalized_data
    return data
