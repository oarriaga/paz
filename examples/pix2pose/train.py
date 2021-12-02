import os
import glob
from tensorflow.keras.optimizers import Adam
from paz.abstract import GeneratingSequence
from paz.models.segmentation import UNET_VGG16

from scenes import PixelMaskRenderer
from pipelines import DomainRandomization
from loss import WeightedReconstruction
from metrics import mean_squared_error as MSE

# global training parameters
H, W, num_channels = image_shape = [128, 128, 3]
beta = 3.0
batch_size = 32
num_classes = 3
learning_rate = 0.001
max_num_epochs = 10
steps_per_epoch = 1000
inputs_to_shape = {'input_1': [H, W, 3]}
labels_to_shape = {'masks': [H, W, 4]}

# global rendering parameters
root_path = os.path.expanduser('~')
background_wildcard = '.keras/paz/datasets/voc-backgrounds/*.png'
background_wildcard = os.path.join(root_path, background_wildcard)
image_paths = glob.glob(background_wildcard)
num_occlusions = 1
viewport_size = image_shape[:2]
light = [1.0, 30]
y_fov = 3.14159 / 4.0

# power drill parameters
"""
OBJ_name = '.keras/paz/datasets/ycb_models/035_power_drill/textured.obj'
distance = [0.3, 0.5]
top_only = False
roll = 3.14159
shift = 0.05
"""

# hammer parameters
OBJ_name = '.keras/paz/datasets/ycb_models/048_hammer/textured.obj'
distance = [0.5, 0.6]
top_only = False
roll = 3.14159
shift = 0.05

path_OBJ = os.path.join(root_path, OBJ_name)

renderer = PixelMaskRenderer(path_OBJ, viewport_size, y_fov, distance,
                             light, top_only, roll, shift)

processor = DomainRandomization(
    renderer, image_shape, image_paths, inputs_to_shape,
    labels_to_shape, num_occlusions)

sequence = GeneratingSequence(processor, batch_size, steps_per_epoch)

weighted_reconstruction = WeightedReconstruction(beta)

model = UNET_VGG16(num_classes, image_shape, freeze_backbone=True)
optimizer = Adam(learning_rate)
model.compile(optimizer, weighted_reconstruction, metrics=MSE)

model.fit(
    sequence,
    epochs=max_num_epochs,
    # callbacks=[stop, log, save, plateau, draw],
    verbose=1,
    workers=0)

model.save_weights('UNET-VGG16_weights_hammer_10.hdf5')
