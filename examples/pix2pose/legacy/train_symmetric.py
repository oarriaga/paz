import os
import glob
import numpy as np
from tensorflow.keras.optimizers import Adam
from paz.abstract import GeneratingSequence
from paz.models.segmentation import UNET_VGG16
from backend import build_rotation_matrix_z

from scenes import PixelMaskRenderer
from pipelines import DomainRandomization
from loss import WeightedSymmetricReconstruction
from loss import WeightedReconstruction
from metrics import mean_squared_error

image_shape = [128, 128, 3]
root_path = os.path.expanduser('~')
background_wildcard = '.keras/paz/datasets/voc-backgrounds/*.png'
background_wildcard = os.path.join(root_path, background_wildcard)
image_paths = glob.glob(background_wildcard)
# path_OBJ = '.keras/paz/datasets/ycb_models/035_power_drill/textured.obj'
path_OBJ = 'single_solar_panel_02.obj'
path_OBJ = os.path.join(root_path, path_OBJ)
num_occlusions = 1
viewport_size = image_shape[:2]
y_fov = 3.14159 / 4.0
distance = [1.0, 1.0]
# distance = [0.3, 0.5]
light = [1.0, 30]
top_only = False
roll = 3.14159
shift = 0.05
batch_size = 32
beta = 3.0
alpha = 0.1
filters = 16
num_classes = 3
learning_rate = 0.0001
max_num_epochs = 5
beta = 10.0
steps_per_epoch = 1000
H, W, num_channels = image_shape = [128, 128, 3]


renderer = PixelMaskRenderer(path_OBJ, viewport_size, y_fov, distance,
                             light, top_only, roll, shift)
# check why this is needed in this object
renderer.scene.ambient_light = [1.0, 1.0, 1.0]
# pose = np.eye(4)
# pose[1, 1] = +np.cos(np.deg2rad(90))
# pose[1, 2] = -np.sin(np.deg2rad(90))
# pose[2, 2] = +np.cos(np.deg2rad(90))
# pose[2, 1] = +np.sin(np.deg2rad(90))
# renderer.scene.set_pose(renderer.mesh, pose)
# renderer.scene.set_pose(renderer.pixel_mesh, pose)
from paz.backend.image import show_image
from backend import rotate_image
for _ in range(0):
    image, alpha, RGB_mask = renderer.render()
    RGB_mask = RGB_mask[..., 0:3]
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


inputs_to_shape = {'input_1': [H, W, num_channels]}
labels_to_shape = {'masks': [H, W, 4]}

processor = DomainRandomization(
    renderer, image_shape, image_paths, inputs_to_shape,
    labels_to_shape, num_occlusions)

sequence = GeneratingSequence(processor, batch_size, steps_per_epoch)

# build all symmetric rotations for solar pannel
angles = np.linspace(0, 2 * np.pi, 7)[:6]
rotations = np.array([build_rotation_matrix_z(angle) for angle in angles])

loss = WeightedSymmetricReconstruction(rotations, beta)
# loss = WeightedReconstruction(beta)

model = UNET_VGG16(num_classes, image_shape, freeze_backbone=False)
optimizer = Adam(learning_rate)

model.compile(optimizer, loss, mean_squared_error)

model.load_weights('UNET_solar_panel_weights_notsym_04.hdf5')
model.fit(
    sequence,
    epochs=max_num_epochs,
    verbose=1,
    workers=0)

model.save_weights('UNET-VGG_solar_panel_weights_not_and_symmetric.hdf5')

