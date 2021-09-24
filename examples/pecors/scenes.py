import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pyrender
import trimesh
import math
import cv2

from paz.backend.render import sample_uniformly, split_alpha_channel
from paz.backend.render import random_perturbation, sample_point_in_sphere
from paz.backend.render import compute_modelview_matrices
from pyrender import PerspectiveCamera, OffscreenRenderer, DirectionalLight
from pyrender import RenderFlags, Mesh, Scene
import trimesh


def draw_circle(image, point, color=(255, 0, 0), radius=5, inner_circle=True):
    """ Draws a circle in image.

    # Arguments
        image: Numpy array of shape ``[H, W, 3]``.
        point: List of length two indicating ``(y, x)``
            openCV coordinates.
        color: List of length three indicating RGB color of point.
        radius: Integer indicating the radius of the point to be drawn.

    # Returns
        Numpy array with shape ``[H, W, 3]``. Image with circle.
    """
    if inner_circle:
        cv2.circle(image, tuple(point), radius, (0, 0, 0), cv2.FILLED)
        inner_radius = int(.8 * radius)
    else:
        inner_radius = radius
    # color = color[::-1]  # transform to BGR for openCV

    cv2.circle(image, tuple(point), inner_radius, tuple(color), cv2.FILLED)
    return image


def quarternion_to_rotation_matrix(q):
    """Transforms quarternion into rotation vector
    # Arguments
        q: quarternion, Numpy array of shape ``[4]``
    # Returns
        Numpy array representing a rotation vector having a shape ``[3]``.
    """
    rotation_matrix = np.array([[1 - 2*(q[1]**2 + q[2]**2), 2*(q[0]*q[1] - q[3]*q[2]), 2*(q[3]*q[1] + q[0]*q[2])],
                                [2*(q[0]*q[1] + q[3]*q[2]), 1 - 2*(q[0]**2 + q[2]**2), 2*(q[1]*q[2] - q[3]*q[0])],
                                [2*(q[0]*q[2] - q[3]*q[1]), 2*(q[3]*q[0] + q[1]*q[2]), 1 - 2*(q[0]**2 + q[1]**2)]])

    return np.squeeze(rotation_matrix)

def rotation_matrix_to_axis_angle(rotation_matrix):
    # Code taken from https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToAngle/index.htm
    epsilon01 = 0.01
    # Distinguish between 0° and 180° (both are singularities)
    epsilon02 = 0.1

    if (abs(rotation_matrix[0, 1] - rotation_matrix[1, 0]) < epsilon01 and
    abs(rotation_matrix[0, 2] - rotation_matrix[2, 0]) < epsilon01 and
    abs(rotation_matrix[1, 2] - rotation_matrix[2, 1]) < epsilon01):

        if (abs(rotation_matrix[0, 1] + rotation_matrix[1, 0]) < epsilon02 and
        abs(rotation_matrix[0, 1] + rotation_matrix[1, 0]) < epsilon02 and
        abs(rotation_matrix[0, 1] + rotation_matrix[1, 0]) < epsilon02 and
        abs(rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2] - 3) < epsilon02):
            # Angle is 0°
            print("0 Grad")
            return np.array([1, 0, 0]), 0

        # Other angle is 180°
        print("180 Grad")
        angle = np.pi
        xx = (rotation_matrix[0, 0] + 1)/2
        yy = (rotation_matrix[1, 1] + 1)/2
        zz = (rotation_matrix[2, 2] + 1)/2
        xy = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / 4
        xz = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / 4
        yz = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / 4

        if (xx > yy) and (xx > zz):
            if xx < epsilon01:
                x = 0
                y = 0.7071
                z = 0.7071
            else:
                x = np.sqrt(xx)
                y = xy/x
                z = xz/x
        elif yy > zz:
            if yy < epsilon01:
                x = 0.7071
                y = 0
                z = 0.7071
            else:
                y = np.sqrt(yy)
                x = xy/y
                z = yz/y
        else:
            if zz < epsilon01:
                x = 0.7071
                y = 0.7071
                z = 0
            else:
                z = np.sqrt(zz)
                x = xz/z
                y = yz/z

        return np.array([x, y, z]), angle

    # No singularities until here, so just normal formula
    s = np.sqrt((rotation_matrix[2, 1] - rotation_matrix[1, 2]) ** 2 +
                (rotation_matrix[0, 2] - rotation_matrix[2, 0]) ** 2 +
                (rotation_matrix[1, 0] - rotation_matrix[0, 1]) ** 2)

    # Prevent division by zero
    if abs(s) < 0.001:
        s = 1

    angle = np.arccos((rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2] - 1) / 2)
    x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
    y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
    z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s

    return np.array([x, y, z]), angle

def rotation_matrix_to_euler_angles(rotation_matrix):
    epsilon = 0.00001

    # Check for one singularity
    if rotation_matrix[1, 0] > (1 - epsilon):
        euler_x = np.arctan2(rotation_matrix[0, 2], rotation_matrix[2, 2])
        euler_y = np.pi/2
        euler_z = 0
        return np.array([euler_x, euler_y, euler_z])

    if (rotation_matrix[1, 0]< -(1 - epsilon)):
        euler_x = np.arctan2(rotation_matrix[0, 2], rotation_matrix[2, 2])
        euler_y = -np.pi/2
        euler_z = 0
        return np.array([euler_x, euler_y, euler_z])

    euler_x = np.arctan2(-rotation_matrix[2, 0], rotation_matrix[0, 0])
    euler_z = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
    euler_y = np.arcsin(rotation_matrix[1, 0])

    return np.array([euler_x, euler_y, euler_z])

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        z = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        x = 0

    return np.array([x, y, z])


def angles_to_color(phi, psi):
    x = np.sin(phi)*np.cos(psi)
    y = np.sin(phi)*np.sin(psi)
    z = np.cos(phi)

    x = (x + 1.) / 2.0
    y = (y + 1.) / 2.0
    z = (z + 1.) / 2.0

    return np.array([x, y, z])

def cartesian_to_color(x, y, z):
    norm = x**2 + y**2

    phi = np.arctan2(z, np.sqrt(norm))
    psi = np.arctan2(y,x)

    return angles_to_color(phi, psi)


def map_to_image_location(point3d, w, h, projection, view):
    # code taken from https://stackoverflow.com/questions/67517809/mapping-3d-vertex-to-pixel-using-pyreder-pyglet-opengl/67534695#67534695
    depth = -(view @ point3d.T)[2]
    p = projection @ view @ point3d.T
    p = p / p[3]
    p[0] = (w / 2 * p[0] + w / 2)
    p[1] = h - (h / 2 * p[1] + h / 2)
    return p[0], p[1], depth


class SingleView():
    """Render-ready scene composed of a single object and a single moving camera.

    # Arguments
        filepath: String containing the path to an OBJ file.
        viewport_size: List, specifying [H, W] of rendered image.
        y_fov: Float indicating the vertical field of view in radians.
        distance: List of floats indicating [max_distance, min_distance]
        light: List of floats indicating [max_light, min_light]
        top_only: Boolean. If True images are only take from the top.
        roll: Float, to sample [-roll, roll] rolls of the Z OpenGL camera axis.
        shift: Float, to sample [-shift, shift] to move in X, Y OpenGL axes.
    """
    def __init__(self, filepath, viewport_size=(128, 128), y_fov=3.14159 / 4.0,
                 camera_distance_bounds=[0.3, 0.5], mesh_translation_bound = 0.05,
                 light_bounds=[0.5, 30], top_only=False,
                 roll=None, shift=None):
        self.camera_distance_bounds, self.roll, self.shift = camera_distance_bounds, roll, shift
        self.light_bounds, self.top_only = light_bounds, top_only
        self._build_scene(filepath, viewport_size, light_bounds, y_fov)
        self.renderer = OffscreenRenderer(viewport_size[0], viewport_size[1])
        self.RGBA = RenderFlags.RGBA
        self.epsilon = 0.01
        self.mesh_translation_bound = mesh_translation_bound
        self.viewport_size = viewport_size

    def _build_scene(self, path, size, light, y_fov):
        self.scene = Scene(bg_color=[0, 0, 0, 0])
        self.light = self.scene.add(DirectionalLight([1.0, 1.0, 1.0], np.mean(light)))
        self.camera = PerspectiveCamera(y_fov, aspectRatio=np.divide(*size))
        self.camera_node = self.scene.add(self.camera)
        self.mesh = self.scene.add(Mesh.from_trimesh(trimesh.load(path), smooth=True))
        self.world_origin = self.mesh.mesh.centroid

    def _sample_parameters(self):
        camera_distance = sample_uniformly(self.camera_distance_bounds)
        light_intensity = sample_uniformly(self.light_bounds)

        mesh_rotation = trimesh.transformations.random_quaternion()
        mesh_rotation = np.array([mesh_rotation[1], mesh_rotation[2], mesh_rotation[3], mesh_rotation[0]])
        mesh_translation = np.array([np.random.uniform(-self.mesh_translation_bound, self.mesh_translation_bound),
                                     np.random.uniform(-self.mesh_translation_bound, self.mesh_translation_bound),
                                     np.random.uniform(-self.mesh_translation_bound, self.mesh_translation_bound)])

        return camera_distance, light_intensity, mesh_rotation, mesh_translation

    def render(self):
        camera_distance, light_intensity, mesh_rotation, mesh_translation = self._sample_parameters()
        camera_origin = np.array([0., 0., camera_distance])

        camera_to_world, world_to_camera = compute_modelview_matrices(camera_origin, self.world_origin, None, None)

        self.scene.set_pose(self.camera_node, camera_to_world)
        self.scene.set_pose(self.light, camera_to_world)
        self.light.light.intensity = light_intensity


        mesh_translation = np.array([0, 0, 0])
        #angle = -np.pi/2
        #mesh_rotation = np.array([np.sin(angle/2), 0, 0, np.cos(angle/2)])

        # Set the mesh rotation and translation
        self.mesh.rotation = mesh_rotation
        self.mesh.translation = mesh_translation

        image_original, depth_original = self.renderer.render(self.scene, flags=pyrender.constants.RenderFlags.RGBA)
        image_original, alpha_original = split_alpha_channel(image_original)

        top_point_3d = quarternion_to_rotation_matrix(mesh_rotation)@(self.mesh.mesh.centroid + np.array([0, self.mesh.mesh.extents[1] / 2, 0])) + mesh_translation
        top_point_3d = np.concatenate((top_point_3d, np.array([1])))
        x_top_point, y_top_point, _ = map_to_image_location(top_point_3d, self.viewport_size[0], self.viewport_size[0], self.camera.get_projection_matrix(*self.viewport_size), world_to_camera)

        # Calculate color for the top point of the object
        top_point_3d_no_translation = quarternion_to_rotation_matrix(mesh_rotation)@(self.mesh.mesh.centroid + np.array([0, self.mesh.mesh.extents[1] / 2, 0]))
        rotation_matrix = quarternion_to_rotation_matrix(mesh_rotation)
        color = top_point_3d_no_translation[:3]/np.linalg.norm(top_point_3d_no_translation[:3])
        color = (color+1)/2.

        image_circle = np.zeros_like(image_original).astype("float")
        image_circle = draw_circle(image_circle, point=(int(x_top_point), int(y_top_point)), color=(color[0], color[1], color[2]), inner_circle=False, radius=5)
        image_circle = (image_circle*255.).astype("uint8")

        # Calculate depth image
        center_point_3d = quarternion_to_rotation_matrix(mesh_rotation)@self.mesh.mesh.centroid + mesh_translation
        center_point_3d = np.concatenate((center_point_3d, np.array([1])))
        x_center_point, y_center_point, depth_center = map_to_image_location(center_point_3d, self.viewport_size[0], self.viewport_size[0], self.camera.get_projection_matrix(*self.viewport_size), world_to_camera)

        image_depth = np.zeros_like(image_original).astype("float")
        image_depth = draw_circle(image_depth, point=(int(x_center_point), int(y_center_point)), color=(depth_center, depth_center, depth_center), inner_circle=False, radius=5)
        image_depth = (image_depth*255.).astype("uint8")

        return image_original, alpha_original, image_circle, image_depth, color


if __name__ == "__main__":
    obj_path = "/home/fabian/.keras/datasets/tless_obj/obj_000003.obj"
    save_path = "/media/fabian/Data/Masterarbeit/data/tless_obj03"
    num_train_images = 100
    num_test_images = 100

    colors = list()
    renderer = SingleView(filepath=obj_path)
    

    for mode, num_images in zip(("train", "test"), (num_train_images, num_test_images)):
        for i in tqdm(range(num_images)):
            image_original, alpha_original, image_circle, image_depth, color = renderer.render()
            colors.append(color)

            np.save(os.path.join(save_path, "{}/image_original/image_original_{}".format(mode, str(i).zfill(7))), image_original)
            np.save(os.path.join(save_path, "{}/alpha_original/alpha_original_{}".format(mode, str(i).zfill(7))), alpha_original)
            np.save(os.path.join(save_path, "{}/image_circle/image_circle_{}".format(mode, str(i).zfill(7))), image_circle)
            np.save(os.path.join(save_path, "{}/image_depth/image_depth_{}".format(mode, str(i).zfill(7))), image_depth)

    """
    for _ in range(10):
        image_original, alpha_original, image_circle, image_depth, color = renderer.render()

        plt.imshow(image_original)
        plt.show()

        plt.imshow(image_circle)
        plt.show()

        plt.imshow(image_depth)
        plt.show()
    """