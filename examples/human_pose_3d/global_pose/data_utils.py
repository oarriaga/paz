"""Utility functions for dealing with human3.6m data."""

import copy
import os
import glob

import numpy as np
import cdflib

import cameras

# Human3.6m IDs for training and testing
TRAIN_SUBJECTS = [1, 5, 6, 7, 8]
TEST_SUBJECTS = [9, 11]

coco_part_labels = [
    'nose', 'eye_l', 'eye_r', 'ear_l', 'ear_r',
    'sho_l', 'sho_r', 'elb_l', 'elb_r', 'wri_l', 'wri_r',
    'hip_l', 'hip_r', 'kne_l', 'kne_r', 'ank_l', 'ank_r'
]

# Joints in H3.6M -- data has 32 joints, but only 17 that move; these are the indices.
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

# Joints in COCO, 2D poses from HigherHRNet --> data has 17 joints; these are the indices. Hip,12,14,16,11,13,15,Spine,Thorax,0,Head,5,7,9,6,8,10
# to make compatible with Human3.6M, Nose -> Neck/Nose; Ankle -> Foot
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


def preprocess_2d_data(poses_2d):
    """Preprocesses 2d detections loaded from text file by creating some extra joints
       and converting COCO joint order to H36M.

    Args
        poses_2d: list of 2d detections obtained from HigherHRNet loaded from a text file
    Returns
        poses: nx32 np array with 2d poses
    """
    # Permutation that goes from COCO detections to H36M ordering.
    COCO_TO_GT_PERM = np.array([COCO_NAMES.index(h) for h in H36M_NAMES if h != '' and h in COCO_NAMES])
    assert np.all(COCO_TO_GT_PERM == np.array([4, 12, 14, 16, 11, 13, 15, 2, 1, 0, 5, 7, 9, 6, 8, 10]))

    poses_2d = np.array(poses_2d)
    poses_2d = poses_2d[:, :, :2]  # Take x and y coord from the HigherHRNet output
    poses_2d = np.reshape(poses_2d,
                          (poses_2d.shape[0], -1))  # reshape to make it compatible with input the model expects

    # make Thorax, mid-point of shoulders i.e. COCO_NAMES[5] & COCO_NAMES[6]
    poses_2d[:, 2:4] = (poses_2d[:, 10:12] + poses_2d[:, 12:14]) / 2

    # make Hip, mid-point of hips i.e. COCO_NAMES[11] & COCO_NAMES[12]
    poses_2d[:, 8:10] = (poses_2d[:, 22:24] + poses_2d[:, 24:26]) / 2

    # make Spine, mid-point of thorax and hip i.e. COCO_NAMES[1] & COCO_NAMES[4]
    poses_2d[:, 4:6] = (poses_2d[:, 2:4] + poses_2d[:, 8:10]) / 2

    # Reshape into (n, 17, 2) matrix
    poses_2d = np.reshape(poses_2d, [poses_2d.shape[0], 17, 2])

    # Permute the loaded data to make it compatible with H36M
    poses = poses_2d[:, COCO_TO_GT_PERM, :]

    # Reshape back into n x (32*2) matrix
    poses = np.reshape(poses, [poses.shape[0], -1])
    return poses


def load_params():
    """Loads normalization statistics: mean and stdev, dimensions used and ignored from npy files

    Returns
        data_mean: nxd np array with the mean of the data
        data_std: nxd np array with the standard deviation of the data
        dim_to_use: nxd np array of dimensions used in the model
        dim_to_ignore: nxd np array of dimensions not used in the model
    """
    path_prefix = os.path.dirname(os.path.abspath(__file__))
    data_mean_2d = np.load(os.path.join(path_prefix, '..', 'files/data_mean_2d.npy'))
    data_std_2d = np.load(os.path.join(path_prefix, '..', 'files/data_std_2d.npy'))
    dim_to_use_2d = np.load(os.path.join(path_prefix, '..', 'files/dim_to_use_2d.npy'))
    dim_to_ignore_2d = np.load(os.path.join(path_prefix, '..', 'files/dim_to_ignore_2d.npy'))
    data_mean_3d = np.load(os.path.join(path_prefix, '..', 'files/data_mean_3d.npy'))
    data_std_3d = np.load(os.path.join(path_prefix, '..', 'files/data_std_3d.npy'))
    dim_to_use_3d = np.load(os.path.join(path_prefix, '..', 'files/dim_to_use_3d.npy'))
    dim_to_ignore_3d = np.load(os.path.join(path_prefix, '..', 'files/dim_to_ignore_3d.npy'))

    return data_mean_2d, data_std_2d, dim_to_use_2d, dim_to_ignore_2d, data_mean_3d, data_std_3d, dim_to_use_3d, dim_to_ignore_3d


def normalize_data(data, data_mean, data_std, dim_to_use):
    """Normalizes a dictionary of poses

    Args
        data: dictionary where values are
        data_mean: np vector with the mean of the data
        data_std: np vector with the standard deviation of the data
        dim_to_use: list of dimensions to keep in the data
    Returns
        data_out: dictionary with same keys as data, but values have been normalized
    """
    data_out = {}

    for key in data.keys():
        data[key] = data[key][:, dim_to_use]
        mu = data_mean[dim_to_use]
        stddev = data_std[dim_to_use]
        data_out[key] = np.divide((data[key] - mu), stddev)

    return data_out


def unNormalizeData(normalized_data, data_mean, data_std, dimensions_to_ignore):
    """Un-normalizes a matrix whose mean has been substracted and that has been divided by
       standard deviation. Some dimensions might also be missing

    Args
        normalized_data: nxd matrix to unnormalize
        data_mean:  nxd np array with the mean of the data
        data_std:  nxd np array with the standard deviation of the data
        dimensions_to_ignore:  nxd np array of dimensions that were removed from the original data
    Returns
        orig_data: the input normalized_data, but unnormalized
    """
    T = normalized_data.shape[0]  # Batch size
    D = data_mean.shape[0]  # Dimensionality; 2d data: 64;  3d data: 96  (32 joints)

    orig_data = np.zeros((T, D), dtype=np.float32)
    dimensions_to_use = np.array([dim for dim in range(D)
                                  if dim not in dimensions_to_ignore])

    orig_data[:, dimensions_to_use] = normalized_data

    stdMat = data_std.reshape((1, D))
    stdMat = np.repeat(stdMat, T, axis=0)
    meanMat = data_mean.reshape((1, D))
    meanMat = np.repeat(meanMat, T, axis=0)
    orig_data = np.multiply(orig_data, stdMat) + meanMat
    return orig_data


def normalization_stats(complete_data, dim, predict_14=False):
    """Computes normalization statistics: mean and stdev, dimensions used and ignored

    Args
        complete_data: nxd np array with poses
        dim. integer={2,3} dimensionality of the data
        predict_14. boolean. Whether to use only 14 joints
    Returns
        data_mean: np vector with the mean of the data
        data_std: np vector with the standard deviation of the data
        dimensions_to_ignore: list of dimensions not used in the model
        dimensions_to_use: list of dimensions used in the model
    """
    if not dim in [2, 3]:
        raise ValueError('dim must be 2 or 3')

    data_mean = np.mean(complete_data, axis=0)
    data_std = np.std(complete_data, axis=0)

    # Encodes which 17 (or 14) 2d-3d pairs we are predicting
    dimensions_to_ignore = []
    if dim == 2:
        dimensions_to_use = np.where(np.array([x != '' and x != 'Neck/Nose' for x in H36M_NAMES]))[0]
        dimensions_to_use = np.sort(np.hstack((dimensions_to_use * 2, dimensions_to_use * 2 + 1)))
        dimensions_to_ignore = np.delete(np.arange(len(H36M_NAMES) * 2), dimensions_to_use)
    else:  # dim == 3
        dimensions_to_use = np.where(np.array([x != '' for x in H36M_NAMES]))[0]
        dimensions_to_use = np.delete(dimensions_to_use, [0, 7, 9] if predict_14 else 0)

        dimensions_to_use = np.sort(np.hstack((dimensions_to_use * 3,
                                               dimensions_to_use * 3 + 1,
                                               dimensions_to_use * 3 + 2)))
        dimensions_to_ignore = np.delete(np.arange(len(H36M_NAMES) * 3), dimensions_to_use)

    return data_mean, data_std, dimensions_to_ignore, dimensions_to_use


def define_actions(action):
    """Given an action string, returns a list of corresponding actions.

    Args
        action: String. either "all" or one of the h36m actions
    Returns
        actions: List of strings. Actions to use.
    Raises
        ValueError: if the action is not a valid action in Human 3.6M
    """
    actions = ["Directions", "Discussion", "Eating", "Greeting",
               "Phoning", "Photo", "Posing", "Purchases",
               "Sitting", "SittingDown", "Smoking", "Waiting",
               "WalkDog", "Walking", "WalkTogether"]

    if action == "All" or action == "all":
        return actions

    if not action in actions:
        raise ValueError("Unrecognized action: %s" % action)

    return [action]


def load_data(bpath, subjects, actions, dim=3):
    """Loads 2d ground truth from disk, and puts it in an easy-to-acess dictionary

    Args
        bpath: String. Path where to load the data from
        subjects: List of integers. Subjects whose data will be loaded
        actions: List of strings. The actions to load
        dim: Integer={2,3}. Load 2 or 3-dimensional data
    Returns:
        data: Dictionary with keys k=(subject, action, seqname)
        values v=(nx(32*2) matrix of 2d ground truth)
        There will be 2 entries per subject/action if loading 3d data
        There will be 8 entries per subject/action if loading 2d data
    """

    if not dim in [2, 3]:
        raise ValueError('dim must be 2 or 3')

    data = {}

    for subj in subjects:
        for action in actions:
            dpath = os.path.join(bpath, 'S{0}'.format(subj), 'MyPoseFeatures/D{0}_Positions'.format(dim),
                                 '{0}*.cdf'.format(action))
            # #print( dpath )
            fnames = glob.glob(dpath)
            loaded_seqs = 0

            for fname in fnames:
                seqname = os.path.basename(fname)

                # This rule makes sure SittingDown is not loaded when Sitting is requested
                if action == "Sitting" and seqname.startswith("SittingDown"):
                    continue

                # This rule makes sure that WalkDog and WalkTogeter are not loaded when
                # Walking is requested.
                if seqname.startswith(action):
                    # #print( fname )
                    loaded_seqs = loaded_seqs + 1

                    cdf_file = cdflib.CDF(fname)
                    poses = cdf_file.varget("Pose").squeeze()
                    cdf_file.close()

                    data[(subj, action, seqname)] = poses

            if dim == 2:
                assert loaded_seqs == 8, "Expecting 8 sequences, found {0} instead".format(loaded_seqs)
            else:
                assert loaded_seqs == 2, "Expecting 2 sequences, found {0} instead".format(loaded_seqs)

    return data


def transform_world_to_camera(poses_set, cams, ncams=4):
    """Project 3d poses from world coordinate to camera coordinate system

    Args
        poses_set: dictionary with 3d poses
        cams: dictionary with cameras
        ncams: number of cameras per subject
    Return:
        t3d_camera: dictionary with 3d poses in camera coordinate
    """
    t3d_camera = {}
    for t3dk in sorted(poses_set.keys()):

        subj, action, seqname = t3dk
        t3d_world = poses_set[t3dk]

        for c in range(ncams):
            R, T, _, _, _, _, name = cams[(subj, c + 1)]
            camera_coord = cameras.world_to_camera_frame(np.reshape(t3d_world, [-1, 3]), R, T)
            camera_coord = np.reshape(camera_coord, [-1, len(H36M_NAMES) * 3])

            sname = seqname[:-3] + name + ".h5"  # e.g.: Waiting 1.58860488.h5
            t3d_camera[(subj, action, sname)] = camera_coord

    return t3d_camera


def read_3d_data(actions, data_dir, camera_frame, rcams, predict_14=False):
    """Loads 3d poses, zero-centres and normalizes them

    Args
        actions: list of strings. Actions to load
        data_dir: string. Directory where the data can be loaded from
        camera_frame: boolean. Whether to convert the data to camera coordinates
        rcams: dictionary with camera parameters
        predict_14: boolean. Whether to predict only 14 joints
    Returns
        train_set: dictionary with loaded 3d poses for training
        test_set: dictionary with loaded 3d poses for testing
        data_mean: vector with the mean of the 3d training data
        data_std: vector with the standard deviation of the 3d training data
        dim_to_ignore: list with the dimensions to not predict
        dim_to_use: list with the dimensions to predict
        train_root_positions: dictionary with the 3d positions of the root in train
        test_root_positions: dictionary with the 3d positions of the root in test
    """
    # Load 3d data
    train_set = load_data(data_dir, TRAIN_SUBJECTS, actions, dim=3)
    test_set = load_data(data_dir, TEST_SUBJECTS, actions, dim=3)

    if camera_frame:
        train_set = transform_world_to_camera(train_set, rcams)
        test_set = transform_world_to_camera(test_set, rcams)

    # Apply 3d post-processing (centering around root)
    train_set, train_root_positions = postprocess_3d(train_set)
    test_set, test_root_positions = postprocess_3d(test_set)

    # Compute normalization statistics
    complete_train = copy.deepcopy(np.vstack(list(train_set.values())))
    data_mean, data_std, dim_to_ignore, dim_to_use = normalization_stats(complete_train, dim=3, predict_14=predict_14)

    # Divide every dimension independently
    train_set = normalize_data(train_set, data_mean, data_std, dim_to_use)
    test_set = normalize_data(test_set, data_mean, data_std, dim_to_use)

    return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use, train_root_positions, test_root_positions


def postprocess_3d(poses_set):
    """Center 3d points around root

    Args
        poses_set: dictionary with 3d data
    Returns
        poses_set: dictionary with 3d data centred around root (center hip) joint
        root_positions: dictionary with the original 3d position of each pose
    """
    root_positions = {}
    for k in poses_set.keys():
        # Keep track of the global position
        root_positions[k] = copy.deepcopy(poses_set[k][:, :3])

        # Remove the root from the 3d position, so that other joints get equal weight and model doesnt focus just on the root
        poses = poses_set[k]
        poses = poses - np.tile(poses[:, :3], [1, len(H36M_NAMES)])
        poses_set[k] = poses

    return poses_set, root_positions


def create_2d_data(actions, data_dir, rcams):
    """Creates 2d poses by projecting 3d poses with the corresponding camera
       parameters. Also normalizes the 2d poses

    Args
        actions: list of strings. Actions to load
        data_dir: string. Directory where the data can be loaded from
        rcams: dictionary with camera parameters
    Returns
        train_set: dictionary with projected 2d poses for training
        test_set: dictionary with projected 2d poses for testing
        data_mean: vector with the mean of the 2d training data
        data_std: vector with the standard deviation of the 2d training data
        dim_to_ignore: list with the dimensions to not predict
        dim_to_use: list with the dimensions to predict
    """

    # Load 3d data
    train_set = load_data(data_dir, TRAIN_SUBJECTS, actions, dim=3)
    test_set = load_data(data_dir, TEST_SUBJECTS, actions, dim=3)

    # Create 2d data by projecting with camera parameters
    train_set = project_to_cameras(train_set, rcams)
    test_set = project_to_cameras(test_set, rcams)

    # Compute normalization statistics.
    complete_train = copy.deepcopy(np.vstack(list(train_set.values())))
    data_mean, data_std, dim_to_ignore, dim_to_use = normalization_stats(complete_train, dim=2)

    # Divide every dimension independently
    train_set = normalize_data(train_set, data_mean, data_std, dim_to_use)
    test_set = normalize_data(test_set, data_mean, data_std, dim_to_use)

    return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use


def project_to_cameras(poses_set, cams, ncams=4):
    """Project 3d poses using camera parameters

    Args
        poses_set: dictionary with 3d poses
        cams: dictionary with camera parameters
        ncams: number of cameras per subject
    Returns
        t2d: dictionary with 2d poses
    """
    t2d = {}

    for t3dk in sorted(poses_set.keys()):
        subj, a, seqname = t3dk
        t3d = poses_set[t3dk]

        for cam in range(ncams):
            R, T, f, c, k, p, name = cams[(subj, cam + 1)]
            pts2d, _, _, _, _ = cameras.project_point_radial(np.reshape(t3d, [-1, 3]), R, T, f, c, k, p)

            pts2d = np.reshape(pts2d, [-1, len(H36M_NAMES) * 2])
            sname = seqname[:-3] + name + ".h5"  # e.g.: Waiting 1.58860488.h5
            t2d[(subj, a, sname)] = pts2d

    return t2d
