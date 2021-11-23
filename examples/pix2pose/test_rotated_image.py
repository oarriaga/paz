import numpy as np
import os
import glob
from paz.backend.image import show_image

from backend import build_rotation_matrix_z, rotate_image
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
renderer.scene.ambient_light = [1.0, 1.0, 1.0]

for _ in range(3):
    image, alpha, RGB_mask = renderer.render()
    RGB_mask = RGB_mask[..., 0:3]
    show_image(image)
    show_image(RGB_mask)
    angles = np.linspace(0, 2 * np.pi, 7)[0:6]
    images = []
    for angle in angles:
        rotation_matrix = build_rotation_matrix_z(angle)
        rotated_image = rotate_image(RGB_mask, rotation_matrix)
        rotated_image = rotated_image.astype('uint8')
        images.append(rotated_image)
    images = np.concatenate(images, axis=1)
    show_image(images)
