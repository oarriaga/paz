import numpy as np
import sys
from PIL import Image
from PIL import ImageDraw
from functools import reduce
import time
from copy import deepcopy
import matplotlib.pyplot as plt
from paz.backend.render import sample_uniformly, split_alpha_channel
from paz.backend.render import random_perturbation, sample_point_in_sphere
from paz.backend.render import compute_modelview_matrices
from pyrender import PerspectiveCamera, OffscreenRenderer, DirectionalLight, OrthographicCamera
from pyrender import RenderFlags, Mesh, Scene
import trimesh
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)


def map_to_image_location(point3d, w, h, projection, view):
    # code taken from https://stackoverflow.com/questions/67517809/mapping-3d-vertex-to-pixel-using-pyreder-pyglet-opengl/67534695#67534695
    depth = -(view@point3d.T)[2]
    p = projection@view@point3d.T
    p = p / p[3]
    p[0] = (w / 2 * p[0] + w / 2)
    p[1] = h - (h / 2 * p[1] + h / 2)
    return (p[0], p[1], depth)


def get_random_translation(translation_bounds=0.1):
    translation = np.array([np.random.uniform(-translation_bounds, translation_bounds),
                            np.random.uniform(-translation_bounds, translation_bounds),
                            np.random.uniform(-translation_bounds, translation_bounds)])
    return translation


def normalize(v):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v/norm


def create_belief_maps(image_size, bounding_box_points, sigma=16):
    """
    Args:
        img: (tuple) size of the image in the format (x, y)
        bounding_box_points: list of points in the form of
                      [num points, 2 (x,y)]
        sigma: (int) size of the belief map point
    return:
        return an array of arrays representing the belief maps
    """
    belief_maps = list()
    max_width_belief_points = int(sigma*2)

    # Shuffle points to make it random which point is removed if two overlap
    #np.random.shuffle(bounding_box_points)

    #print("Bounding boxes: {}".format(bounding_box_points))

    for point in bounding_box_points:
        belief_map = np.zeros(image_size)

        # Only add a belief point if it is completely inside the image
        if point[0] - max_width_belief_points >= 0 and point[0] + max_width_belief_points < image_size[0] and \
                point[1] - max_width_belief_points >= 0 and point[1] + max_width_belief_points < image_size[1]:

            #print("Point: {}".format(point))
            # Check if there is already a point inside the area where we want to add a new point
            #if belief_maps:
            #    sum_belief_maps = np.sum(np.array(belief_maps), axis=0)
            #    if np.sum(sum_belief_maps[int(point[1]) - max_width_belief_points:int(point[1]) + max_width_belief_points,\
            #                              int(point[0]) - max_width_belief_points:int(point[0]) + max_width_belief_points]) > 0:
            #        belief_maps.append(belief_map)
            #        continue

            # Assign a value to a pixel inside of the belief point depending on how far away it
            # is from the center of the point
            for i in range(int(point[0]) - max_width_belief_points, int(point[0]) + max_width_belief_points):
                for j in range(int(point[1]) - max_width_belief_points, int(point[1]) + max_width_belief_points):
                    belief_map[j, i] = np.exp(-((i - int(point[0]))**2 + (j - int(point[1]))**2) / (2 * (sigma**2)))

            # Normalize belief map so that the sum of all elements is 1
            #belief_map /= np.sum(belief_map)

        belief_maps.append(belief_map)

    return np.asarray(belief_maps)


def create_affinity_maps(image_size, object_center, bounding_box_points, radius=7, sigma=16):
    """
    Args:
        img: (tuple) size of the image in the format (x, y)
        bounding_box_points: list of points in the form of
                      [num points, 2 (x,y)]
        sigma: (int) size of the belief map point
    return:
        return an array of arrays representing the belief maps
    """
    affinity_maps = list()

    # Iterate over all bounding box points
    for bounding_box_point in bounding_box_points:
        affinity_map = Image.new("L", image_size, "black")

        # Draw an ellipse at the bounding box location
        draw = ImageDraw.Draw(affinity_map)
        draw.ellipse((bounding_box_point[0] - radius, bounding_box_point[1] - radius,
                      bounding_box_point[0] + radius, bounding_box_point[1] + radius), 1)

        del draw

        array = np.array(affinity_map)

        # Calculate distance to the center
        center_vector = np.array(object_center) - np.array(bounding_box_point)
        center_vector = normalize(center_vector)

        # Create affinity maps for the x and y direction

        # Normalize affinity maps
        xvec = center_vector[0]
        yvec = center_vector[1]

        norms = np.sqrt(xvec * xvec + yvec * yvec)

        center_vector[0] /= norms
        center_vector[1] /= norms

        affinity_maps.append(array * center_vector[0])
        affinity_maps.append(array * center_vector[1])

    return np.asarray(affinity_maps)


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
    def __init__(self, filepath, colors, viewport_size=(128, 128), y_fov=3.14159 / 4.0,
                 distance=[0.9, 1.5], light_bounds=[0.5, 30], top_only=False,
                 roll=None, shift=None, scaling_factor=8.0):
        self.distance, self.roll, self.shift = distance, roll, shift
        self.light_intensity, self.top_only = light_bounds, top_only
        self.RGBA = RenderFlags.RGBA
        self.epsilon = 0.01
        self.viewport_size = viewport_size
        self.colors = colors
        self.meshes_original = list()
        self.meshes_ambient_light = list()
        self.mesh_origins = list()

        self.world_origin = np.array([0, 0, 0])
        self.scaling_factor = scaling_factor

        self._build_scene(filepath, viewport_size, light_bounds, y_fov)
        self.renderer = OffscreenRenderer(self.viewport_size[0], self.viewport_size[1])

    def _build_scene(self, paths, size, light, y_fov, rotation_matrix=np.eye(4), translation=np.zeros(3)):
        # Load the object
        loaded_trimeshes = [trimesh.load(path) for path in paths]

        # First scene = original scene
        self.scene_original = Scene(bg_color=[0, 0, 0, 0])
        self.light_original = self.scene_original.add(DirectionalLight([1.0, 1.0, 1.0], np.mean(light)))
        #self.camera = PerspectiveCamera(y_fov, aspectRatio=np.divide(*size))
        self.camera = OrthographicCamera(0.3, 0.3)
        self.camera_original = self.scene_original.add(self.camera)

        for loaded_trimesh in loaded_trimeshes:
            self.color_mesh_uniform(loaded_trimesh, np.array([255, 0, 0]))
            mesh_original = self.scene_original.add(Mesh.from_trimesh(loaded_trimesh, smooth=True))
            self.meshes_original.append(mesh_original)
            self.mesh_origins.append(mesh_original.mesh.centroid)

        #self.world_origin = mesh_original.mesh.centroid

        # Second scene = scene with ambient light
        """
        loaded_trimeshes = [trimesh.load(path) for path in paths]
        self.scene_ambient_light = Scene(bg_color=[0, 0, 0, 0], ambient_light=[255, 255, 255])
        #self.camera = PerspectiveCamera(y_fov, aspectRatio=np.divide(*size))
        self.camera = OrthographicCamera(0.3, 0.3)
        self.camera_ambient_light = self.scene_ambient_light.add(self.camera)

        for loaded_trimesh, color in zip(loaded_trimeshes, self.colors):
            self.color_mesh_uniform(loaded_trimesh, np.array(color))
            mesh_ambient_light = self.scene_ambient_light.add(Mesh.from_trimesh(loaded_trimesh, smooth=True))
            self.meshes_ambient_light.append(mesh_ambient_light)

        #print("World origin: {}".format(self.world_origin))
        #self.sphere = trimesh.creation.icosphere(radius=0.01, color=(0, 255, 0))
        #self.sphere = Mesh.from_trimesh(self.sphere, smooth=False)
        #self.sphere = self.scene_original.add(self.sphere)

        camera_to_world, world_to_camera = compute_modelview_matrices(np.array([.4, .4, .4]), self.world_origin, self.roll, self.shift)
        self.light_original.light.intensity = 5.0
        self.scene_original.set_pose(self.camera_original, camera_to_world)
        self.scene_original.set_pose(self.light_original, camera_to_world)

        self.scene_ambient_light.set_pose(self.camera_ambient_light, camera_to_world)
        """

    def _sample_parameters(self):
        distance = sample_uniformly(self.distance)
        print("Distance: " + str(distance))
        camera_origin = sample_point_in_sphere(distance, self.top_only)
        camera_origin = random_perturbation(camera_origin, self.epsilon)
        #print("Camera origin: " + str(camera_origin))
        return camera_origin

    def render(self):
        start = time.time()

        camera_origin = self._sample_parameters()
        camera_to_world, world_to_camera = compute_modelview_matrices(
            camera_origin, self.world_origin, self.roll, self.shift)
        self.scene_original.set_pose(self.camera_original, camera_to_world)
        self.scene_original.set_pose(self.light_original, camera_to_world)
        #self.scene_ambient_light.set_pose(self.camera_ambient_light, camera_to_world)

        positions, object_centers, extents = list(), list(), list()

        for mesh_original, mesh_origin in zip(self.meshes_original, self.mesh_origins):

            translation = get_random_translation()
            mesh_original.translation = translation
            positions.append(np.append(mesh_origin + translation, [1]))

            print("Center in camera coords (scene): " + str(world_to_camera@np.append(translation + mesh_origin, [1])))

            extents.append(mesh_original.mesh.extents)

        num_objects = len(positions)

        # Stores the bounding box point locations. Format:
        # (number of objects in the scene, number of edges of the bounding box, pixel coordinates of each point)
        bounding_box_points = np.zeros((num_objects, 9, 2))
        bounding_box_points_3d = np.zeros((num_objects, 9, 3))

        # Calculate all the bounding box points
        # Important: the first point in the list is the object center!
        for i, (position, extent) in enumerate(zip(positions, extents)):
            (x, y, depth) = map_to_image_location(position, self.viewport_size[0], self.viewport_size[0], self.camera.get_projection_matrix(128, 128), world_to_camera)
            bounding_box_points[i, 0] = np.array([x, y])
            bounding_box_points_3d[i, 0] = position[:3]

            num_bounding_box_point = 1
            for x_extent in [extent[0]/2, -extent[0]/2]:
                for y_extent in [extent[1]/2, -extent[1]/2]:
                    for z_extent in [extent[2]/2, -extent[2]/2]:
                        point_position = np.array([position[0] + x_extent, position[1] + y_extent, position[2] + z_extent, 1])
                        bounding_box_points_3d[i, num_bounding_box_point] = point_position[:3]

                        (x, y, depth) = map_to_image_location(point_position, self.viewport_size[0], self.viewport_size[0], self.camera.get_projection_matrix(128, 128), world_to_camera)
                        bounding_box_points[i, num_bounding_box_point] = np.array([x, y])
                        num_bounding_box_point += 1

        image_original, depth_original = self.renderer.render(self.scene_original, flags=self.RGBA)
        image_original, alpha_original = split_alpha_channel(image_original)

        self.renderer.delete()

        scaled_viewport_size = (int(self.viewport_size[0]/self.scaling_factor), int(self.viewport_size[1]/self.scaling_factor))
        # Calculate the belief maps
        # Format: (num objects, num bounding box points, image width, image height)
        belief_maps = np.zeros((num_objects, 9, scaled_viewport_size[0], scaled_viewport_size[1]))
        for num_object in range(num_objects):
            belief_maps[num_object] = create_belief_maps(scaled_viewport_size, bounding_box_points[num_object]/self.scaling_factor, sigma=2)

        # Calculate the affinity maps
        # Format: (num objects, num bounding box edges, image width, image height)
        affinity_maps = list()
        """
        affinity_maps = np.zeros((num_objects, 16, scaled_viewport_size[0], scaled_viewport_size[1]))
        for num_object in range(num_objects):
            affinity_maps[num_object] = create_affinity_maps(scaled_viewport_size, bounding_box_points[num_object, 0]/self.scaling_factor, bounding_box_points[num_object, 1:]/self.scaling_factor, radius=1, sigma=1)
        """

        return image_original, alpha_original, bounding_box_points, belief_maps, affinity_maps, bounding_box_points_3d

    def color_mesh_uniform(self, mesh, color):
        vertices = mesh.vertices
        colors = np.tile(color, (len(vertices), 1))

        mesh.visual = mesh.visual.to_color()
        mesh.visual.vertex_colors = colors

        return mesh


if __name__ == "__main__":
    num_samples = 5
    file_paths = ["/home/fabian/.keras/datasets/035_power_drill/tsdf/textured.obj"]#, "/home/fabian/.keras/datasets/011_banana/tsdf/textured.obj"]
    colors = [np.array([255, 0, 0]), np.array([0, 255, 0])]
    viewport_size = (400, 400)

    view = SingleView(filepath=file_paths, colors=colors, viewport_size=viewport_size, scaling_factor=8.0)

    image_original, alpha_original, bounding_box_points, belief_maps, affinity_maps, _ = view.render()

    f, axs = plt.subplots(1, 2)
    #print(affinity_maps.shape)

    axs[0].imshow(image_original)
    axs[1].imshow(np.sum(belief_maps[0, :], axis=0))
    #axs[2].imshow(affinity_maps[0, 1])

    for ax in axs:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    plt.show()
