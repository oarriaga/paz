import numpy as np
import sys
import time
from copy import deepcopy
import matplotlib.pyplot as plt
from paz.backend.render import sample_uniformly, split_alpha_channel
from paz.backend.render import random_perturbation, sample_point_in_sphere
from paz.backend.render import compute_modelview_matrices
from paz.backend.quaternion import quarternion_to_rotation_matrix, rotation_vector_to_quaternion, quaternion_multiply, quaternion_to_euler
from pyrender import PerspectiveCamera, OffscreenRenderer, DirectionalLight
from pyrender import RenderFlags, Mesh, Scene, Material
import pyrender
import trimesh
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D


#fig = plt.figure()
#ax = Axes3D(fig)

np.set_printoptions(threshold=sys.maxsize)


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
                 distance=[0.3, 0.5], light_bounds=[0.5, 30], top_only=False,
                 roll=None, shift=None):
        self.distance, self.roll, self.shift = distance, roll, shift
        self.light_intensity, self.top_only = light_bounds, top_only
        self._build_scene(filepath, viewport_size, light_bounds, y_fov, colors=False)
        #self.renderer = OffscreenRenderer(viewport_size[0], viewport_size[1])
        self.RGBA = RenderFlags.RGBA
        self.epsilon = 0.01
        self.viewport_size = viewport_size

    def _build_scene(self, path, size, light, y_fov, colors=True, rotation_matrix=np.eye(4), translation=np.zeros(3)):
        # Load the object
        loaded_trimesh = trimesh.load(path)
        # Create two scenes: one for the colored objcet one for the error object
        self.scene_original = Scene(bg_color=[0, 0, 0, 0])
        light_original = self.scene_original.add(DirectionalLight([1.0, 1.0, 1.0], np.mean(light)))
        camera = self.scene_original.add(PerspectiveCamera(y_fov, aspectRatio=np.divide(*size)))
        self.color_mesh_uniform(loaded_trimesh, np.array([255, 0, 0]))
        self.mesh_original = self.scene_original.add(Mesh.from_trimesh(loaded_trimesh, smooth=True))
        self.world_origin = self.mesh_original.mesh.centroid
        camera_to_world, world_to_camera = compute_modelview_matrices(np.array([.4, .4, .4]), self.world_origin, self.roll, self.shift)
        light_original.light.intensity = 5.0
        self.scene_original.set_pose(camera, camera_to_world)
        self.scene_original.set_pose(light_original, camera_to_world)

        loaded_trimesh = trimesh.load(path)
        self.scene_color = Scene(bg_color=[0, 0, 0, 0], ambient_light=[1.0, 1.0, 1.0, 1.0])
        #light_color.light.intensity = 30.
        camera = self.scene_color.add(PerspectiveCamera(y_fov, aspectRatio=np.divide(*size)))
        self.color_mesh(loaded_trimesh)
        #self.color_mesh_uniform(loaded_trimesh, np.array([30, 0, 255]))
        self.mesh_color = self.scene_color.add(Mesh.from_trimesh(loaded_trimesh, smooth=False))
        self.world_origin = self.mesh_color.mesh.centroid
        self.scene_color.set_pose(camera, camera_to_world)
        #self.scene_color.set_pose(light_color, camera_to_world)

        #print(camera_to_world)
        #self.cylinder = self.scene_original.add(Mesh.from_trimesh(trimesh.creation.cylinder(radius=0.01, height=0.5)))
        #self.sphere = self.scene_original.add(Mesh.from_trimesh(trimesh.creation.icosphere(subdivisions=3, radius=0.05)))

    def render(self):
        self.renderer = OffscreenRenderer(self.viewport_size[0], self.viewport_size[1])
        rotation = trimesh.transformations.random_quaternion()

        angle = np.pi
        vertical_rotation = np.array([0, np.sin(angle / 2), 0, np.cos(angle / 2)])

        # Combine the random rotation with the symmetry rotation
        #final_rotation = quaternion_multiply(rotation, vertical_rotation)
        #final_rotation /= np.linalg.norm(final_rotation)

        self.mesh_original.rotation = rotation
        self.mesh_color.rotation = rotation

        image_original, depth_original = self.renderer.render(self.scene_original, flags=self.RGBA)
        image_original, alpha_original = split_alpha_channel(image_original)

        image_colors, _ = self.renderer.render(self.scene_color, flags=pyrender.constants.RenderFlags.FLAT)
        self.renderer.delete()

        return image_original, image_colors, alpha_original, None

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
        color_copy = deepcopy(colors)
        # Rotate the object
        angle = np.pi
        for i in range(colors.shape[0]):
            colors[i] = quarternion_to_rotation_matrix(np.array([0, np.sin(angle / 2), 0, np.cos(angle / 2)]))@colors[i]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(colors[:5000, 0]/255., colors[:5000, 1]/255., colors[:5000, 2]/255., c=color_copy[:5000]/255.)
        plt.show()

        mesh.visual = mesh.visual.to_color()
        mesh.visual.vertex_colors = colors

        return mesh

    def color_mesh_uniform(self, mesh, color):
        vertices = mesh.vertices
        colors = np.tile(color, (len(vertices), 1))

        mesh.visual = mesh.visual.to_color()
        mesh.visual.vertex_colors = colors

        return mesh

    def color_mesh_uniform_vertices(self, mesh, color):
        vertices = mesh.vertices
        colors = np.tile(color, (len(vertices), 1))

        mesh.visual = mesh.visual.to_color()
        mesh.visual.vertex_colors = colors

        return vertices, colors


if __name__ == "__main__":
    num_samples = 5
    list_images = list()
    #view = SingleView(filepath="/home/fabian/.keras/datasets/036_wood_block/textured_edited.obj")
    view = SingleView(filepath="/home/fabian/.keras/datasets/001_chips_can/tsdf/textured_edited.obj")

    for _ in range(num_samples):
        image_original, image_colors, alpha_original, rotation = view.render()
        list_images.append((image_original, image_colors))

    fig = plt.figure(constrained_layout=False)
    spec = gridspec.GridSpec(ncols=num_samples, nrows=2, figure=fig)
    for i in range(num_samples):
        sub_fig = fig.add_subplot(spec[0, i])
        sub_fig.imshow(list_images[i][0])

        sub_fig = fig.add_subplot(spec[1, i])
        sub_fig.imshow(list_images[i][1])

    plt.show()