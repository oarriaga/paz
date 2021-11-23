import numpy as np
import os
import glob
from paz.backend.image import show_image

from backend import build_rotation_matrix_z
from backend import normalized_device_coordinates_to_image
from backend import image_to_normalized_device_coordinates
from scenes import PixelMaskRenderer

scale = 4
image_shape = [128 * scale, 128 * scale, 3]
root_path = os.path.expanduser('~')
background_wildcard = '.keras/paz/datasets/voc-backgrounds/*.png'
background_wildcard = os.path.join(root_path, background_wildcard)
image_paths = glob.glob(background_wildcard)

path_OBJ = 'single_solar_panel_02.obj'
path_OBJ = os.path.join(root_path, path_OBJ)
num_occlusions = 1
viewport_size = image_shape[:2]
y_fov = 3.14159 / 4.0
distance = [1.0, 1.0]
light = [1.0, 30]
top_only = False
roll = 3.14159
shift = 0.05

renderer = PixelMaskRenderer(path_OBJ, viewport_size, y_fov, distance,
                             light, top_only, roll, shift)


def rotate_image(image, rotation_matrix, epsilon=1e-4):
    mask_image = np.sum(image, axis=-1, keepdims=True)
    mask_image = mask_image != 0

    image = image_to_normalized_device_coordinates(image)
    # image = image / 255.0
    print(image.min(), image.max())
    # image = (image * 2) - 1

    # rotated_image = image + epsilon
    rotated_image = np.einsum('ij,klj->kli', rotation_matrix, image)
    rotated_image = normalized_device_coordinates_to_image(rotated_image)
    # rotated_image = (rotated_image + 1) / 2
    # print(rotated_image.min(), rotated_image.max())

    # rotated_image = np.clip(rotated_image, a_min=0.0, a_max=1.0)
    rotated_image = np.clip(rotated_image, a_min=0.0, a_max=255.0)
    # rotated_image = rotated_image * 255.0
    rotated_image = rotated_image * mask_image
    return rotated_image


image, alpha, RGB_mask = renderer.render()
RGB_mask = RGB_mask[..., 0:3]
show_image(image)
show_image(RGB_mask)
angles = np.linspace(0, 2 * np.pi, 7)
images = []
for angle in angles:
    print('-' * 40)
    print('angle', angle)
    rotation_matrix = build_rotation_matrix_z(angle)
    print(rotation_matrix)
    rotated_image = rotate_image(RGB_mask, rotation_matrix)
    rotated_image = rotated_image.astype('uint8')
    images.append(rotated_image)
    # show_image(rotated_image)
images = np.concatenate(images, axis=1)
show_image(images)
