import os
import glob

import numpy as np
from tensorflow.keras.optimizers import Adam
from paz.backend.image import show_image
from paz.models import UNET_VGG16
from paz.abstract.sequence import GeneratingSequence
from paz.backend.render import compute_modelview_matrices

from scenes import CanonicalScene
from backend import build_rotation_matrix_z
from backend import build_rotation_matrix_x
from backend import build_rotation_matrix_y
from backend import to_affine_matrix
from pipelines import DomainRandomization
from loss import WeightedReconstruction
from metrics import mean_squared_error


root_path = os.path.expanduser('~')
num_occlusions = 1
image_shape = (128, 128, 3)
viewport_size = image_shape[:2]
y_fov = 3.14159 / 4.0
light = [1.0, 30]

# training parameters
H, W, num_channels = image_shape
batch_size = 32
steps_per_epoch = 1000
beta = 3.0
num_classes = 3
learning_rate = 0.001
max_num_epochs = 5

"""
path_OBJ = 'single_solar_panel_02.obj'
angles = np.linspace(0, 2 * np.pi, 7)[:6]
symmetric_rotations = np.array(
    [build_rotation_matrix_z(angle) for angle in angles])
min_corner = [0.0, 0.0, -0.4]
max_corner = [0.0, 0.0, +0.0]
camera_rotation = build_rotation_matrix_x(np.pi)
translation = np.array([0.0, 0.0, -1.0])
camera_pose = to_affine_matrix(camera_rotation, translation)
"""

# large clamp parameters
# REMEMBER TO CHANGE THE Ns coefficient to values between [0, 1] in
# textured.mtl. For example change 96.07 to .967
OBJ_name = '.keras/paz/datasets/ycb_models/051_large_clamp/textured.obj'
translation = np.array([0.0, 0.0, 0.25])
camera_pose, y = compute_modelview_matrices(translation, np.zeros((3)))
align_z = build_rotation_matrix_z(np.pi / 20)
camera_pose[:3, :3] = np.matmul(align_z, camera_pose[:3, :3])
min_corner = [-0.05, -0.02, -0.05]
max_corner = [+0.05, +0.02, +0.01]

angles = [0.0, np.pi]
symmetric_rotations = np.array(
    [build_rotation_matrix_y(angle) for angle in angles])


path_OBJ = os.path.join(root_path, OBJ_name)
renderer = CanonicalScene(path_OBJ, camera_pose, min_corner,
                          max_corner, symmetric_rotations)
renderer.scene.ambient_light = [1.0, 1.0, 1.0]
image = renderer.render_symmetries()
show_image(image)

background_wildcard = '.keras/paz/datasets/voc-backgrounds/*.png'
background_wildcard = os.path.join(root_path, background_wildcard)
image_paths = glob.glob(background_wildcard)

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
model.fit(
    sequence,
    epochs=max_num_epochs,
    verbose=1,
    workers=0)
model.save_weights('UNET-VGG_large_clamp_canonical.hdf5')
