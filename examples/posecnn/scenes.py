import numpy as np
import sys
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
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

#fig = plt.figure()
#ax = Axes3D(fig)

np.set_printoptions(threshold=sys.maxsize)


def map_to_image_location(point3d, w, h, projection, view):
    # code taken from https://stackoverflow.com/questions/67517809/mapping-3d-vertex-to-pixel-using-pyreder-pyglet-opengl/67534695#67534695
    depth = -(view@point3d.T)[2]
    p = projection@view@point3d.T
    p = p / p[3]
    p[0] = (w / 2 * p[0] + w / 2)
    p[1] = h - (h / 2 * p[1] + h / 2)
    return (p[0], p[1], depth)


def get_random_translation(translation_bounds=0.2):
    translation = np.array([np.random.uniform(-translation_bounds, translation_bounds),
                            np.random.uniform(-translation_bounds, translation_bounds),
                            np.random.uniform(-translation_bounds, translation_bounds)])
    return translation


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

    for point in bounding_box_points:
        belief_map = np.zeros(image_size)
        # Only add a belief point if it is completely inside the image
        if point[0] - max_width_belief_points >= 0 and point[0] + max_width_belief_points < image_size[0] and \
                point[1] - max_width_belief_points >= 0 and point[1] + max_width_belief_points < image_size[1]:
            # Assign a value to a pixel inside of the belief point depending on how far away it
            # is from the center of the point
            for i in range(int(point[0]) - max_width_belief_points, int(point[0]) + max_width_belief_points):
                for j in range(int(point[1]) - max_width_belief_points, int(point[1]) + max_width_belief_points):
                    belief_map[i, j] = np.exp(-((i - point[0])**2 + (j - point[1])**2) / (2 * (sigma**2)))

        belief_maps.append(belief_map)

    return np.asarray(belief_maps)

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
                 distance=[0.5, 0.9], light_bounds=[0.5, 30], top_only=False,
                 roll=None, shift=None):
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

        self._build_scene(filepath, viewport_size, light_bounds, y_fov)

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

    def _sample_parameters(self):
        distance = sample_uniformly(self.distance)
        camera_origin = sample_point_in_sphere(distance, self.top_only)
        camera_origin = random_perturbation(camera_origin, self.epsilon)
        return camera_origin

    def render(self):
        start = time.time()
        self.renderer = OffscreenRenderer(self.viewport_size[0], self.viewport_size[1])

        camera_origin = self._sample_parameters()
        camera_to_world, world_to_camera = compute_modelview_matrices(
            camera_origin, self.world_origin, self.roll, self.shift)
        self.scene_original.set_pose(self.camera_original, camera_to_world)
        self.scene_ambient_light.set_pose(self.camera_ambient_light, camera_to_world)

        positions, object_centers, extents = list(), list(), list()

        for mesh_original, mesh_ambient_light, mesh_origin in zip(self.meshes_original, self.meshes_ambient_light, self.mesh_origins):
            translation = get_random_translation()
            mesh_original.translation = translation
            mesh_ambient_light.translation = translation
            positions.append(np.append(mesh_origin + translation, [1]))
            extents.append(mesh_original.mesh.extents)

        num_objects = len(positions)

        # Stores the bounding box point locations. Format:
        # (number of objects in the scene, number of edges of the bounding box, pixel coordinates of each point)
        bounding_box_points = np.zeros((num_objects, 9, 2))

        # Calculate all the bounding box points
        # Important: the first point in the list is the object center!
        for i, (position, extent) in enumerate(zip(positions, extents)):
            (x, y, depth) = map_to_image_location(position, self.viewport_size[0], self.viewport_size[0], self.camera.get_projection_matrix(128, 128), world_to_camera)
            #object_centers.append(np.array([x, y, depth]))
            bounding_box_points[i, 0] = np.array([x, y])

            num_bounding_box_point = 1
            for x_extent in [extent[0]/2, -extent[0]/2]:
                for y_extent in [extent[1]/2, -extent[1]/2]:
                    for z_extent in [extent[2]/2, -extent[2]/2]:
                        point_position = np.array([position[0] + x_extent, position[1] + y_extent, position[2] + z_extent, 1])

                        (x, y, depth) = map_to_image_location(point_position, self.viewport_size[0], self.viewport_size[0], self.camera.get_projection_matrix(128, 128), world_to_camera)
                        bounding_box_points[i, num_bounding_box_point] = np.array([x, y])
                        num_bounding_box_point += 1

        image_original, depth_original = self.renderer.render(self.scene_original, flags=self.RGBA)
        image_original, alpha_original = split_alpha_channel(image_original)

        image_ambient_light, _ = self.renderer.render(self.scene_ambient_light, flags=self.RGBA)
        image_ambient_light, _ = split_alpha_channel(image_ambient_light)

        #for point in bounding_box_points[0]:
        #    image_original = image_original.copy()
        #    image_original[int(point[1]), int(point[0])] = np.array([255, 255, 255])

        self.renderer.delete()

        # Calculate the belief maps
        # Format: (num objects, num bounding box edges, image width, image height)
        belief_maps = np.zeros((num_objects, 9, self.viewport_size[0], self.viewport_size[1]))
        for num_object in range(num_objects):
            bm = create_belief_maps(self.viewport_size, bounding_box_points[num_object])
            belief_maps[num_object] = create_belief_maps(self.viewport_size, bounding_box_points[num_object], sigma=5)

        plt.imshow(belief_maps[0, 0])
        plt.show()

        """
        # Generate the semantic segmentation image
        image_masks, image_masks_3d = list(), list()

        for color in self.colors:
            image_mask = np.apply_along_axis(lambda x: int(np.array_equal(x, color)), 2, image_ambient_light)
            image_masks.append(image_mask)
            image_masks_3d.append(np.repeat(image_mask[:, :, np.newaxis], 3, axis=2))

        distances_x_direction, distances_y_direction, depth_images = list(), list(), list()
        # Generate the images with centers in x and y direction
        for image_mask, object_center in zip(image_masks, object_centers):
            image_pixel_values = np.flip(np.array(np.meshgrid(np.arange(0, self.viewport_size[0], 1), np.arange(0, self.viewport_size[1], 1))).T, axis=2)
            image_pixel_values = image_pixel_values*np.reshape(image_mask, (self.viewport_size[0], self.viewport_size[1], 1))
            distance_x_direction = np.apply_along_axis(lambda x: (x[0] - object_center[0])/np.linalg.norm(x - object_center[:2]), 2, image_pixel_values)*image_mask
            distance_y_direction = np.apply_along_axis(lambda x: (x[1] - object_center[1]) / np.linalg.norm(x - object_center[:2]), 2, image_pixel_values)*image_mask

            # Generate depth image
            depth_image = image_mask*object_center[2]

            distances_x_direction.append(distance_x_direction)
            distances_y_direction.append(distance_y_direction)
            depth_images.append(depth_image)

        distance_x_direction = reduce(lambda x, y: x + y, distances_x_direction)
        distance_y_direction = reduce(lambda x, y: x + y, distances_y_direction)
        depth_image = reduce(lambda x, y: x + y, depth_images)
        """

        return image_original, alpha_original, bounding_box_points#, object_centers, image_masks, distance_x_direction, distance_y_direction, depth_image

    def normalize(self, x, x_min, x_max):
        return (x-x_min)/(x_max-x_min)

    def color_mesh(self, mesh):
        """ color the mesh
        # Arguments
            mesh: obj mesh
        # Returns
            mesh: colored obj mesh
        """
        vertices = mesh.vertices
        x_min = mesh.vertices[:, 0].min()
        x_max = mesh.vertices[:, 0].max()
        y_min = mesh.vertices[:, 1].min()
        y_max = mesh.vertices[:, 1].max()
        z_min = mesh.vertices[:, 2].min()
        z_max = mesh.vertices[:, 2].max()

        # make vertices using RGB format
        vertices_x = 255 * self.normalize(mesh.vertices[:, 0:1], x_min, x_max)
        vertices_y = 255 * self.normalize(mesh.vertices[:, 1:2], y_min, y_max)
        vertices_z = 255 * self.normalize(mesh.vertices[:, 2:3], z_min, z_max)

        vertices_x = vertices_x.astype('uint8')
        vertices_y = vertices_y.astype('uint8')
        vertices_z = vertices_z.astype('uint8')
        colors = np.hstack([vertices_x, vertices_y, vertices_z])

        mesh.visual = mesh.visual.to_color()
        mesh.visual.vertex_colors = colors

        return mesh

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
    view = SingleView(filepath=file_paths, colors=colors, viewport_size=(224, 224))

    image_original, alpha_original, bounding_box_points, object_centers, semantic_segmentation_images, distance_x_direction, distance_y_direction, depth_image = view.render()

    f, axs = plt.subplots(1, 5)
    #plt.axis('off')

    axs[0].imshow(image_original)
    axs[1].imshow(semantic_segmentation_images[0])
    axs[2].imshow(distance_x_direction)
    axs[3].imshow(distance_y_direction)
    axs[4].imshow(depth_image)

    for ax in axs:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    plt.show()
