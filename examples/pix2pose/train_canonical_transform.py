import os
import glob

import numpy as np
from tensorflow.keras.optimizers import Adam
from paz.backend.image import show_image
from paz.models import UNET_VGG16
from paz.abstract.sequence import GeneratingSequence

from scenes import CanonicalScene
from backend import build_rotation_matrix_z
from backend import build_rotation_matrix_x
from backend import to_affine_matrix
from pipelines import DomainRandomization
from loss import WeightedReconstruction
from metrics import mean_squared_error


path_OBJ = 'single_solar_panel_02.obj'
root_path = os.path.expanduser('~')
path_OBJ = os.path.join(root_path, path_OBJ)
num_occlusions = 1
image_shape = (128, 128, 3)
viewport_size = image_shape[:2]
y_fov = 3.14159 / 4.0
distance = [1.0, 1.0]
light = [1.0, 30]

angles = np.linspace(0, 2 * np.pi, 7)[:6]
symmetric_rotations = np.array(
    [build_rotation_matrix_z(angle) for angle in angles])
min_corner = [0.0, 0.0, -0.4]
max_corner = [0.0, 0.0, +0.0]
camera_rotation = build_rotation_matrix_x(np.pi)
translation = np.array([0.0, 0.0, -1.0])
camera_pose = to_affine_matrix(camera_rotation, translation)
renderer = CanonicalScene(path_OBJ, camera_pose, min_corner,
                          max_corner, symmetric_rotations)
# from pyrender import Viewer
# Viewer(scene.scene)
renderer.scene.ambient_light = [1.0, 1.0, 1.0]
image = renderer.render_symmetries()
show_image(image)
for _ in range(100):
    image, alpha, RGB_mask = renderer.render()
    show_image(image)
    show_image(RGB_mask[:, :, 0:3])

background_wildcard = '.keras/paz/datasets/voc-backgrounds/*.png'
background_wildcard = os.path.join(root_path, background_wildcard)
image_paths = glob.glob(background_wildcard)

H, W, num_channels = image_shape
batch_size = 32
steps_per_epoch = 1000
beta = 3.0
num_classes = 3
learning_rate = 0.001
max_num_epochs = 5

inputs_to_shape = {'input_1': [H, W, num_channels]}
labels_to_shape = {'masks': [H, W, 4]}

processor = DomainRandomization(
    renderer, image_shape, image_paths, inputs_to_shape,
    labels_to_shape, num_occlusions)

sequence = GeneratingSequence(processor, batch_size, steps_per_epoch)

# build all symmetric rotations for solar pannel
angles = np.linspace(0, 2 * np.pi, 7)[:6]
rotations = np.array([build_rotation_matrix_z(angle) for angle in angles])

loss = WeightedReconstruction(beta)

model = UNET_VGG16(num_classes, image_shape, freeze_backbone=True)
optimizer = Adam(learning_rate)

model.compile(optimizer, loss, mean_squared_error)
"""
model.fit(
    sequence,
    epochs=max_num_epochs,
    verbose=1,
    workers=0)
model.save_weights('UNET-VGG_solar_panel_canonical.hdf5')
"""
