import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

from paz.backend.render import sample_uniformly, split_alpha_channel
from paz.backend.render import random_perturbation, sample_point_in_sphere
from paz.backend.render import compute_modelview_matrices
import pyrender
from pyrender import PerspectiveCamera, OffscreenRenderer, DirectionalLight
from pyrender import RenderFlags, Mesh, Scene
import trimesh

epos_colormap = np.asarray([
  [0, 0, 0], [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
  [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
  [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7], [150, 5, 61],
  [120, 120, 70], [8, 255, 51], [255, 6, 82], [143, 255, 140], [204, 255, 4],
  [255, 51, 7], [204, 70, 3], [0, 102, 200], [61, 230, 250], [255, 6, 51],
  [11, 102, 255], [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
  [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224], [255, 184, 6],
  [10, 255, 71], [255, 41, 10], [7, 255, 255], [224, 255, 8], [102, 8, 255],
  [255, 61, 6], [255, 194, 7], [255, 122, 8], [0, 255, 20], [255, 8, 41],
  [255, 5, 153], [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
  [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0], [255, 31, 0],
  [255, 224, 0], [153, 255, 0], [0, 0, 255], [255, 71, 0], [0, 235, 255],
  [0, 173, 255], [31, 0, 255], [11, 200, 200], [255, 82, 0], [0, 255, 245],
  [0, 61, 255], [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
  [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0], [0, 82, 255],
  [0, 255, 41], [0, 255, 173], [10, 0, 255], [173, 255, 0], [0, 255, 153],
  [255, 92, 0], [255, 0, 255], [255, 0, 245], [255, 0, 102], [255, 173, 0],
  [255, 0, 20], [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
  [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255], [0, 112, 255],
  [51, 0, 255], [0, 194, 255], [0, 122, 255], [0, 255, 163], [255, 153, 0],
  [0, 255, 10], [255, 112, 0], [143, 255, 0], [82, 0, 255], [163, 255, 0],
  [255, 235, 0], [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
  [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112], [92, 255, 0],
  [0, 224, 255], [112, 224, 255], [70, 184, 160], [163, 0, 255],
  [153, 0, 255], [71, 255, 0], [255, 0, 163], [255, 204, 0], [255, 0, 143],
  [0, 255, 235], [133, 255, 0], [255, 0, 235], [245, 0, 255], [255, 0, 122],
  [255, 245, 0], [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
  [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204], [41, 0, 255],
  [41, 255, 0], [173, 0, 255], [0, 245, 255], [71, 0, 255], [122, 0, 255],
  [0, 255, 184], [0, 92, 255], [184, 255, 0], [0, 133, 255], [255, 214, 0],
  [25, 194, 194], [102, 255, 0], [92, 0, 255]
])

class Fragment():
    def __init__(self):
        self.center = (0, 0, 0)
        # List of tuples. Each tuple has the format (3D point coords, 2D pixel value)
        self.points2D = list()
        self.points3D = list()
        self.points3D_fragment_coords= list()
        self.norm_value = 1

    def calculate_fragment_coordinates(self):
        points2D = np.asarray(self.points2D)
        points3D = np.asarray(self.points3D)
        
        # Calculate the norm value first (largest distance in any direction, i.e. x, y, or z)
        fragment_extent = [np.max(points3D[:, dim]) - np.min(points3D[:, dim]) for dim in [0, 1, 2]]
        self.norm_value = max(fragment_extent)
        
        # Calculate 3D points in fragment coordinates
        for point3D in points3D:
            point3d_fragment_coords = (point3D - self.center)/self.norm_value
            point3d_fragment_coords = (point3d_fragment_coords+1)*127.5
            self.points3D_fragment_coords.append(point3d_fragment_coords)


def calculate_fragment_centers(vertices, num_fragments=80):
    # Choose a random point as the first fragment center
    fragment_centers = [np.squeeze(np.asarray(vertices[np.random.randint(vertices.shape[0], size=1), :]))]

    # Iterate num_fragments times overs vertices to find the fragment centers
    for i in range(num_fragments-1):
        min_distance_value = 0
        min_distance_point = None

        for vertex in tqdm(vertices):
            distance_values = list()
            for fragment_center in fragment_centers:
                distance_values.append(np.linalg.norm(fragment_center - vertex))
                
            if min(distance_values) > min_distance_value:
                min_distance_value = min(distance_values)
                min_distance_point = vertex
                
        fragment_centers.append(np.asarray(min_distance_point))

    np.save("./fragment_centers/fragment_centers_tless_obj05.npy", fragment_centers)


def create_fragments(fragment_centers, vertices):
    list_vertices_to_fragment_centers = list()
    fragment_list = list()

    # Create all the required fragments
    for fragment_center in fragment_centers:
        fragment = Fragment()
        fragment.center = fragment_center
        fragment_list.append(fragment)

    # Find the closest fragment center for each vertex
    for vertex in tqdm(vertices):
        closest_fragment_center_distance = np.finfo('d').max
        closest_fragment_center = -1

        for i, fragment_center in enumerate(fragment_centers):
            if np.linalg.norm(fragment_center - vertex) < closest_fragment_center_distance:
                closest_fragment_center_distance = np.linalg.norm(fragment_center - vertex)
                closest_fragment_center = i

        fragment_list[i].points.append(np.asarray(vertex))
        list_vertices_to_fragment_centers.append(closest_fragment_center)

    return list_vertices_to_fragment_centers, fragment_list


def normalize(x, x_min, x_max):
    """
    Normalization for coloring
    """
    return (x - x_min) / (x_max - x_min)


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
    def __init__(self, filepath, path_fragment_centers, viewport_size=(128, 128), y_fov=3.14159 / 4.0,
                 distance=[0.3, 0.5], light=[0.5, 30], top_only=False,
                 roll=None, shift=None):
        self.distance, self.roll, self.shift = distance, roll, shift
        self.light_intensity, self.top_only = light, top_only
        self.fragment_centers = np.load(path_fragment_centers)
        self._build_scene(filepath, viewport_size, light, y_fov)
        self.renderer = OffscreenRenderer(viewport_size[0], viewport_size[1])
        self.RGBA = RenderFlags.RGBA
        self.epsilon = 0.01
        self.viewport_size = viewport_size

    def _build_scene(self, path, size, light, y_fov):
        # Scene with the original image
        self.scene_original = Scene(bg_color=[0, 0, 0, 0])
        self.light_original = self.scene_original.add(DirectionalLight([1.0, 1.0, 1.0], np.mean(light)))
        self.camera_original = self.scene_original.add(PerspectiveCamera(y_fov, aspectRatio=np.divide(*size)))

        loaded_trimesh_original = trimesh.load(path)
        self.mesh_original = self.scene_original.add(Mesh.from_trimesh(loaded_trimesh_original, smooth=True))

        # Scene with the color image
        self.scene = Scene(bg_color=[0, 0, 0, 0])
        self.light = self.scene.add(DirectionalLight([1.0, 1.0, 1.0], np.mean(light)))
        self.camera = self.scene.add(PerspectiveCamera(y_fov, aspectRatio=np.divide(*size)))

        loaded_trimesh = trimesh.load(path)
        self.color_mesh(mesh=loaded_trimesh)
        self.mesh = self.scene.add(Mesh.from_trimesh(loaded_trimesh, smooth=True))
        self.world_origin = self.mesh.mesh.centroid

        self.world_origin = self.mesh.mesh.centroid

    def _sample_parameters(self):
        distance = sample_uniformly(self.distance)
        camera_origin = sample_point_in_sphere(distance, self.top_only)
        camera_origin = random_perturbation(camera_origin, self.epsilon)
        light_intensity = sample_uniformly(self.light_intensity)
        return camera_origin, light_intensity

    def render(self):
        camera_origin, intensity = self._sample_parameters()
        camera_to_world, world_to_camera = compute_modelview_matrices(
            camera_origin, self.world_origin, self.roll, self.shift)
        self.light.light.intensity = intensity

        self.scene_original.set_pose(self.camera_original, camera_to_world)
        self.scene_original.set_pose(self.light_original, camera_to_world)
        self.scene.set_pose(self.camera, camera_to_world)
        self.scene.set_pose(self.light, camera_to_world)

        image_original, _ = self.renderer.render(self.scene_original, flags=pyrender.constants.RenderFlags.RGBA)

        image, depth = self.renderer.render(self.scene, flags=pyrender.constants.RenderFlags.FLAT)
        fragment_image, fragment_list = self.color_fragments_from_image(image, self.mesh.mesh.extents)
        fragment_coords_image = self.color_fragment_coordinates_from_image(image, fragment_list)
        
        mask_object = np.ma.masked_not_equal(np.sum(image, axis=-1), 0.).mask.astype(float)
        mask_object_3d = mask_object[..., np.newaxis]
        image_uniform = np.tile(mask_object_3d, 3)*np.array([255., 0., 0])
        image_uniform = image_uniform.astype("uint8")

        # Create the final output. The output has 4n + 2 channels (n = number of fragments)
        # 2: background or object
        # n: probability that the current pixel belongs to fragment i
        # 3n: fragment relatice coordinates for this pixel in a specific fragment
        epos_output = np.zeros((self.viewport_size[0], self.viewport_size[1], 4*len(self.fragment_centers) + 2))

        # 1. Back ground or specific object?
        epos_output[:, :, 0] = (np.ones_like(mask_object) - mask_object)
        epos_output[:, :, 1] = mask_object

        # 2. Which fragment does the pixel belong to?
        for i, fragment in enumerate(fragment_list):
            for point2D in fragment.points2D:
                epos_output[point2D[0], point2D[1], 2 + i] = 1.

        # 3. What are the fragment coordinates for this pixel?
        for i, fragment in enumerate(fragment_list):
            for point2D, point3D_fragment_coords in zip(fragment.points2D, fragment.points3D_fragment_coords):
                epos_output[point2D[0], point2D[1], len(self.fragment_centers)-1 + 3*i:len(self.fragment_centers)-1 + 3*i + 3] = point3D_fragment_coords

        print(epos_output.shape)
        #image, alpha = split_alpha_channel(image)
        return image_original, image_uniform, fragment_image, fragment_coords_image


    def color_mesh(self, mesh):
        """ color the mesh
        # Arguments
            mesh: obj mesh
        # Returns
            mesh: colored obj mesh
        """
        x_min = mesh.vertices[:, 0].min()
        x_max = mesh.vertices[:, 0].max()
        y_min = mesh.vertices[:, 1].min()
        y_max = mesh.vertices[:, 1].max()
        z_min = mesh.vertices[:, 2].min()
        z_max = mesh.vertices[:, 2].max()

        # make vertices using RGB format
        vertices_x = 255 * normalize(mesh.vertices[:, 0:1], x_min, x_max)
        vertices_y = 255 * normalize(mesh.vertices[:, 1:2], y_min, y_max)
        vertices_z = 255 * normalize(mesh.vertices[:, 2:3], z_min, z_max)

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

    def color_fragments(self, mesh, list_vertices_to_fragment_centers):
        # Create colors for the different fragments
        dict_fragment_center_to_color = dict()
        for i in range(len(list(set(list_vertices_to_fragment_centers)))):
            dict_fragment_center_to_color[i] = epos_colormap[i]#np.array([random.randint(0, 200), random.randint(0, 200), random.randint(0, 200)])

        # Assign the right color to every vertex
        vertices_colors = list()
        for vertex_to_fragment_center in list_vertices_to_fragment_centers:
            vertices_colors.append(dict_fragment_center_to_color[vertex_to_fragment_center])

        vertices_colors = np.vstack(vertices_colors)

        mesh.visual = mesh.visual.to_color()
        mesh.visual.vertex_colors = vertices_colors

        return mesh

    def color_fragments_from_image(self, color_image, object_extent):
        fragment_image = deepcopy(color_image)
        fragment_list = list()
        dict_fragment_center_to_color = dict()

        object_mask = np.ma.masked_not_equal(np.sum(fragment_image, axis=-1), 0.).mask.astype(float)
        object_mask = object_mask[..., np.newaxis]
        object_mask = np.tile(object_mask, 3)

        fragment_image = (fragment_image.astype(float)/127.5)-1.
        fragment_image = fragment_image*(object_extent/2)
        fragment_image = fragment_image*object_mask

        # Define the colors for the fragment centers. The color is scaled accordignly to the 3D position
        # of the fragment center
        for i, fragment_center in enumerate(self.fragment_centers):
            fragment_color = fragment_center/(object_extent/2)
            # Clip the values so that the lowest value is 0.2. If it is 0, some
            # fragments are nearly completely black
            fragment_color = np.clip(((fragment_color + 1)/2), 0.2, 1)* 255.
            fragment_color = fragment_color.astype("uint8")
            dict_fragment_center_to_color[i] = fragment_color
            
            fragment = Fragment()
            fragment.center = fragment_center
            fragment_list.append(fragment)

        # Iterate over all pixels of the color image, translate pixel values back to 3D positions
        # and change the color according to the closest fragment center
        for i in tqdm(range(fragment_image.shape[0])):
            for j in range(fragment_image.shape[1]):

                if np.array_equal(fragment_image[i, j], np.array([0., 0., 0.])):
                    continue

                closest_fragment_center = -1
                closest_fragment_center_distance = np.finfo('d').max

                for num_frag_center, fragment_center in enumerate(self.fragment_centers):
                    if np.linalg.norm(fragment_image[i, j] - fragment_center) < closest_fragment_center_distance:
                        closest_fragment_center_distance = np.linalg.norm(fragment_image[i, j] - fragment_center)
                        closest_fragment_center = num_frag_center

                fragment_list[closest_fragment_center].points3D.append(deepcopy(fragment_image[i, j]))
                fragment_list[closest_fragment_center].points2D.append(np.array([i, j]))

                fragment_image[i, j] = dict_fragment_center_to_color[closest_fragment_center]

        fragment_image = fragment_image.astype("uint8")
        return fragment_image, fragment_list
    
    def color_fragment_coordinates_from_image(self, color_image, fragment_list):
        fragment_coords_color_image = deepcopy(color_image)
        
        for fragment in fragment_list:
            if len(fragment.points2D) == 0:
                continue
            fragment.calculate_fragment_coordinates()
            for point2D, point3D in zip(fragment.points2D, fragment.points3D_fragment_coords):
                fragment_coords_color_image[point2D[0], point2D[1]] = point3D
                
        return fragment_coords_color_image
                
        

if __name__ == "__main__":
    num_samples = 5
    fig, axs = plt.subplots(4, num_samples)

    for i in range(num_samples):
        renderer = SingleView("/home/fabian/.keras/datasets/tless_obj/obj_000005.obj",
                              "/home/fabian/Uni/masterarbeit/src/paz/examples/epos/fragment_centers/fragment_centers_tless_obj05_80.npy",
                              viewport_size=(128, 128))
        image_original, image_uniform, fragment_image, fragment_coords_image = renderer.render()

        axs[0, i].imshow(image_original)
        axs[0, i].get_xaxis().set_visible(False)
        axs[0, i].get_yaxis().set_visible(False)

        axs[1, i].imshow(image_uniform)
        axs[1, i].get_xaxis().set_visible(False)
        axs[1, i].get_yaxis().set_visible(False)

        axs[2, i].imshow(fragment_image)
        axs[2, i].get_xaxis().set_visible(False)
        axs[2, i].get_yaxis().set_visible(False)

        axs[3, i].imshow(fragment_coords_image)
        axs[3, i].get_xaxis().set_visible(False)
        axs[3, i].get_yaxis().set_visible(False)

    plt.show()


