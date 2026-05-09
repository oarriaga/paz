import numpy as np


def proj_3d_to_2d(P, f, c):
    """
    Project points in camera frame from 3d to 2d using intrinsic matrix of the camera

    Args
        P: Nx3 points in camera coordinates
        f: (scalar) Camera focal length
        c: 2x1 image center
    Returns
        Nx2 points in pixel space
    """
    assert len(P.shape) == 2
    assert P.shape[1] == 3

    z = P[:, 2]
    x = (f / z) * P[:, 0] + c[0, 0]
    y = (f / z) * P[:, 1] + c[0, 1]
    return np.column_stack((x, y))


def s2d(poses2d):
    """Computes sum of bone lengths in 2d

    Args
        poses2d: np array of poses in 2d

    Returns
        sum_bl: sum of length of all bones in the 2d skeleton
    """
    assert poses2d[0].shape == (32,), "channels should have 32 entries, it has %d instead" % poses2d[0].shape
    sum_bl = np.zeros(poses2d.shape[0])  # sum of bone lengths, each entry is for each person
    poses2d = np.reshape(poses2d, (poses2d.shape[0], 16, -1))

    start_joints = np.array([1, 2, 3, 1, 5, 6, 1, 8, 9, 9, 11, 12, 9, 14, 15]) - 1
    end_joints = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]) - 1

    for idx, person in enumerate(poses2d):
        for i in np.arange(len(start_joints)):
            bone_length = np.linalg.norm(person[start_joints[i]] - person[end_joints[i]])
            sum_bl[idx] += bone_length
    return sum_bl


def s3d(poses3d):
    """Computes sum of bone lengths in 3d

    Args
        poses3d: np array of predicted poses in 3d

    Returns
        sum_bl: sum of length of all bones in the 3d skeleton
    """
    sum_bl = np.zeros(poses3d.shape[0])  # sum of bone lengths, each entry is for each person
    poses3d = np.reshape(poses3d, (poses3d.shape[0], 16, -1))

    # start_joints = np.array([1, 2, 3, 1, 7, 8, 1, 13, 14, 15, 14, 18, 19, 14, 26, 27]) - 1
    # end_joints = np.array([2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 26, 27, 28]) - 1
    #TODO: CHECK THIS PART THE INDICES!!!!
    start_joints = np.array([1, 2, 3, 1, 5, 6, 1, 8, 9, 9, 11, 12, 9, 14, 15]) - 1
    end_joints = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]) - 1

    for idx, person in enumerate(poses3d):
        for i in np.arange(len(start_joints)):
            bone_length = np.linalg.norm(person[start_joints[i]] - person[end_joints[i]])
            sum_bl[idx] += bone_length

    return sum_bl


def init_translation(f, root_2d, img_center, s2d, s3d):
    """Computes initial 3d translation of root joint

    Args
        f: focal length of the camera in pixels
        root_2d: 2d root joint from HigherHRNet
        img_center: center of the image (or principal point)
        s2d: sum of bone lengths of 2d skeleton
        s3d: sum of bone lengths of 3d skeleton (or can be of it's orthographic projection)

    Returns
        Array of initial estimate of the global position of the root joint in 3d
    """
    ratio = s3d / s2d
    ox, oy = img_center[0]
    Z = f * ratio  # depth coord
    X = (root_2d[:, 0] - ox) * ratio  # horz coord
    Y = (root_2d[:, 1] - oy) * ratio  # vert coord
    return np.column_stack((X, Y, Z)) 


def orthographic_proj(P):
    """Computes orthographic projection of 3d pose

    Args
        P: 3d pose

    Returns
        Array containing the X and Y coordinate of 3d pose
    """
    assert len(P.shape) == 2
    assert P.shape[1] == 3

    x = P[:, 0]
    y = P[:, 1]
    return np.column_stack((x, y))


def pix_to_cam(point, f, c):
    point = (point - c) / f
    z = 2.8  # f in mm for ZED camera, replace with focal length of the camera used
    return np.insert(point, 2, z, axis=1)
