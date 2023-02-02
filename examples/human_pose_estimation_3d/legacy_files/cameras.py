"""Utilities to deal with the cameras of human3.6m"""

from xml.dom import minidom
import numpy as np
from paz.backend.groups.SO3 import build_rotation_matrix_x, \
    build_rotation_matrix_y, build_rotation_matrix_z

CAMERA_ID_TO_NAME = {
    1: "54138969",
    2: "55011271",
    3: "58860488",
    4: "60457274",
}


def get_radial_per_point(camera_radial_distortion, radius_squared, N):
    """ Project points from 3d to 2d using camera parameters
     including radial and tangential distortion

    # Arguments
    camera_radial_distortion:
    radius_squared:1xN squared radius of the projected points before distortion
    N: length of points
    
    #Returns
    radial_per_point: 1xN radial distortion per point
    """
    radius = [radius_squared, radius_squared ** 2, radius_squared ** 3]
    radial_per_point = 1 + np.einsum('ij,ij->j',
                                     np.tile(camera_radial_distortion, (1, N)),
                                     np.array(radius))
    return radial_per_point


def get_project_points(radial_per_point, radius, tangential_distortion,
                       camera_tangential_distortion, focal_point,
                       camera_center):
    XXX = radius * np.tile(radial_per_point + tangential_distortion,
                           (2, 1)) + np.outer(np.array(
        [camera_tangential_distortion[1],
         camera_tangential_distortion[0]]).reshape(-1), radius ** 2)
    Projection = (focal_point * XXX) + camera_center
    Projection = Projection.T
    return Projection


def project_point_radial(points, rotation, translation, focal_point,
                         camera_center,
                         camera_radial_distortion,
                         camera_tangential_distortion):
    """ Project points from 3d to 2d using camera parameters
     including radial and tangential distortion

    # Arguments
    points: Nx3 points in world coordinates
    rotation: 3x3 Camera rotation matrix
    translation: 3x1 Camera translation parameters
    focal_point: (scalar) Camera focal length
    camera_center: 2x1 Camera center
    camera_radial_distortion: 3x1 Camera radial distortion coefficients
    camera_tangential_distortion: 2x1 Camera tangential distortion coefficients
    
    # Returns
    Proj: Nx2 points in pixel space
    depth: 1xN depth of each point in camera space
    radial_per_point: 1xN radial distortion per point
    tangential distortion: 1xN tangential distortion per point
    radius_squared: 1xN squared radius of the projected points before distortion
    """

    N = len(points)
    transformed_pts = rotation.dot(points.T - translation)
    radius = transformed_pts[:2, :] / transformed_pts[2, :]
    radius_squared = radius[0, :] ** 2 + radius[1, :] ** 2
    radial_per_point = get_radial_per_point(camera_radial_distortion,
                                            radius_squared, N)
    tangential_distortion = camera_tangential_distortion[0] * radius[1, :] + \
                            camera_tangential_distortion[1] * radius[0, :]
    projection = get_project_points(radial_per_point, radius,
                                    tangential_distortion,
                                    camera_tangential_distortion, focal_point,
                                    camera_center)
    depth = transformed_pts[2,]
    return projection, depth, radial_per_point, tangential_distortion, radius_squared


def world_to_camera_frame(points, rotation, translation):
    """
    Convert points from world to camera coordinates

    # Arguments
      points: Nx3 3d points in world coordinates
      rotation: 3x3 Camera rotation matrix
      translation: 3x1 Camera translation parameters

    # Returns
      X_cam: Nx3 3d points in camera coordinates
    """
    world_to_cam = rotation.dot(points.T - translation)
    return world_to_cam.T


def camera_to_world_frame(points, rotation, translation):
    """Inverse of world_to_camera_frame

    # Arguments
      P: Nx3 points in camera coordinates
      R: 3x3 Camera rotation matrix
      T: 3x1 Camera translation parameters

    # Returns
      X_cam: Nx3 points in world coordinates
    """
    cam_to_world = rotation.T.dot(
        points.T) + translation  # rotate and translate
    return cam_to_world.T


def get_rotation(w1):
    """Load h36m camera parameters

    # Arguments
      w1: array read from XML metadata
   
    # Returns
      rotation: 3x3 Camera rotation matrix
    """
    rotation_x = build_rotation_matrix_x(w1[0])
    rotation_y = build_rotation_matrix_y(w1[1])
    rotation_z = build_rotation_matrix_z(w1[2])
    rotation = rotation_x.dot(rotation_y).dot(rotation_z)
    return rotation


def load_camera_params(w0, subject, camera):
    """Load h36m camera parameters

    # Arguments
      w0: 300-long array read from XML metadata
      subect: int subject id
      camera: int camera id

    # Returns
      rotation: 3x3 Camera rotation matrix
      translation: 3x1 Camera translation parameters
      focal_length: (scalar) Camera focal length
      camera_center: 2x1 Camera center
      camera_radial_distortion: 3x1 Camera radial distortion coefficients
      camera_tangential_distortion: 2x1 Camera tangential distortion coefficients
      camera_id: String with camera id
    """
    w1 = np.zeros(15)
    start = 6 * ((camera - 1) * 11 + (subject - 1))
    w1[:6] = w0[start:start + 6]
    w1[6:] = w0[(265 + (camera - 1) * 9 - 1): (264 + camera * 9)]
    rotation = get_rotation(w1)
    translation = w1[3:6][:, np.newaxis]
    focal_length = w1[6:8][:, np.newaxis]
    camera_center = w1[8:10][:, np.newaxis]
    camera_radial_distortion = w1[10:13][:, np.newaxis]
    camera_tangential_distortion = w1[13:15][:, np.newaxis]
    camera_id = CAMERA_ID_TO_NAME[camera]
    return rotation, translation, focal_length, camera_center, camera_radial_distortion, camera_tangential_distortion, camera_id


def load_cameras(path_h36m_data, subjects=[1, 5, 6, 7, 8, 9, 11]):
    """Loads the cameras of h36m

    # Arguments
      path_h36m_data: path to xml file with h36m camera data
      subjects: List of ints representing the subject IDs for which cameras are requested

    # Returns
      camera_params_dict: dictionary of 4 tuples per subject ID containing its camera parameters for the 4 h36m cams
    """
    camera_params_dict = {}
    xmldoc = minidom.parse(path_h36m_data)
    string_of_numbers = xmldoc.getElementsByTagName('w0')[0].firstChild.data[
                        1:-1]
    w0 = np.array(list(map(float, string_of_numbers.split(" "))))
    for subject in subjects:
        for camera in range(4):  # There are 4 cameras in human3.6m
            camera_params_dict[(subject, camera + 1)] = load_camera_params(w0,
                                                                           subject,
                                                                           camera + 1)
    return camera_params_dict
