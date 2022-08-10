import os
import glob
import json
import argparse
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import get_file
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, CSVLogger, ModelCheckpoint, ReduceLROnPlateau)

from paz.abstract import GeneratingSequence
from paz.models.segmentation import UNET_VGG16
from paz.optimization.callbacks import DrawInferences
from paz.backend.camera import Camera
from paz.backend.image import write_image
from paz.optimization.losses import WeightedReconstruction
from paz.pipelines.pose import RGBMaskToPose6D
from pipelines import SingleInferencePIX2POSE6D

from scenes import PixelMaskRenderer
from pipelines import DomainRandomization

OBJ_FILE = 'textured.obj'
cache_subdir = 'paz/datasets/ycb_video/035_power_drill'
URL = 'https://github.com/oarriaga/altamira-data/releases/download/v0.12/'
OBJ_FILEPATH = get_file(OBJ_FILE, URL + OBJ_FILE, cache_subdir=cache_subdir)

root_path = os.path.expanduser('~')

description = 'Training script for pix2pose model'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--obj_path', default=OBJ_FILEPATH, type=str,
                    help='Path to OBJ model')
parser.add_argument('--save_path', default='experiments', type=str,
                    help='Path for saving evaluations')
parser.add_argument('--model', default='UNET_VGG16', type=str,
                    choices=['UNET_VGG16'])
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size used during optimization')
parser.add_argument('--learning_rate', default=0.001, type=float,
                    help='Initial learning rate for Adam')
parser.add_argument('--beta', default=3.0, type=float,
                    help='Loss Weight for pixels in object')
parser.add_argument('--max_num_epochs', default=100, type=int,
                    help='Number of epochs before finishing')
parser.add_argument('--steps_per_epoch', default=250, type=int,
                    help='Steps per epoch')
parser.add_argument('--stop_patience', default=5, type=int,
                    help='Early stop patience')
parser.add_argument('--reduce_patience', default=2, type=int,
                    help='Reduce learning rate patience')
parser.add_argument('--run_label', default='RUN_00', type=str,
                    help='Label used to distinguish between different runs')
parser.add_argument('--time', type=str,
                    default=datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
parser.add_argument('--light', nargs='+', type=float, default=[1.0, 30])
parser.add_argument('--y_fov', default=3.14159 / 4.0, type=float,
                    help='Field of view angle in radians')
parser.add_argument('--distance', nargs='+', type=float, default=[0.3, 0.5],
                    help='Distance from camera to origin in meters')
parser.add_argument('--top_only', default=0, choices=[0, 1], type=int,
                    help='Flag for full sphere or top half for rendering')
parser.add_argument('--roll', default=3.14159, type=float,
                    help='Threshold for camera roll in radians')
parser.add_argument('--shift', default=0.05, type=float,
                    help='Threshold of random shift of camera')
parser.add_argument('--num_occlusions', default=1, type=int,
                    help='Number of occlusions added to image')
parser.add_argument('--num_test_images', default=100, type=int,
                    help='Number of test images')
parser.add_argument('--image_size', default=128, type=int,
                    help='Size of the side of a square image e.g. 64')
parser.add_argument('--background_wildcard', type=str,
                    help='Wildcard for backgroun images', default=os.path.join(
                        root_path,
                        '.keras/paz/datasets/voc-backgrounds/*.png'))
args = parser.parse_args()


# loading background image paths
image_paths = glob.glob(args.background_wildcard)
if len(image_paths) == 0:
    raise ValueError('Background images not found. Provide path to png images')

# setting rendering function
H, W, num_channels = image_shape = [args.image_size, args.image_size, 3]
renderer = PixelMaskRenderer(
    args.obj_path, [H, W], args.y_fov, args.distance, args.light,
    args.top_only, args.roll, args.shift)

# building full processor
inputs_to_shape = {'input_1': [H, W, num_channels]}    # inputs RGB
labels_to_shape = {'masks': [H, W, num_channels + 1]}  # labels RGBMask + alpha
processor = DomainRandomization(
    renderer, image_shape, image_paths, inputs_to_shape,
    labels_to_shape, args.num_occlusions)


# building python generator
sequence = GeneratingSequence(processor, args.batch_size, args.steps_per_epoch)


# metric for labels with alpha mask
def mean_squared_error(y_true, y_pred):
    squared_difference = tf.square(y_true[:, :, :, 0:3] - y_pred[:, :, :, 0:3])
    return tf.reduce_mean(squared_difference, axis=-1)


# instantiating the model and loss
model = UNET_VGG16(num_channels, image_shape, freeze_backbone=True)
optimizer = Adam(args.learning_rate)
loss = WeightedReconstruction(args.beta)
model.compile(optimizer, loss, mean_squared_error)

# building experiment path
experiment_label = '_'.join([model.name, args.run_label, args.time])
experiment_path = os.path.join(args.save_path, experiment_label)

# setting additional callbacks
log = CSVLogger(os.path.join(experiment_path, 'optimization.log'))
stop = EarlyStopping('loss', patience=args.stop_patience, verbose=1)
plateau = ReduceLROnPlateau('loss', patience=args.reduce_patience, verbose=1)
save_filename = os.path.join(experiment_path, 'model_weights.hdf5')
save = ModelCheckpoint(save_filename, 'loss', verbose=1, save_best_only=True,
                       save_weights_only=True)

image_directory = os.path.join(experiment_path, 'original_images')
if not os.path.exists(image_directory):
    os.makedirs(image_directory)

images = []
for image_arg in range(args.num_test_images):
    image, alpha, masks = renderer.render()
    image = np.copy(image)  # TODO: renderer outputs unwritable numpy arrays
    masks = np.copy(masks)  # TODO: renderer outputs unwritable numpy arrays
    image_filename = 'image_%03d.png' % image_arg
    masks_filename = 'masks_%03d.png' % image_arg
    image_directory = os.path.join(experiment_path, 'original_images')
    image_filename = os.path.join(image_directory, image_filename)
    masks_filename = os.path.join(image_directory, masks_filename)
    write_image(image_filename, image)
    write_image(masks_filename, masks)
    images.append(image)

# setting drawing callback
camera = Camera()
camera.distortion = np.zeros((4))
camera.intrinsics_from_HFOV(image_shape=(args.image_size, args.image_size))
object_sizes = renderer.mesh.mesh.extents * 100  # from meters to milimiters
# camera.intrinsics = renderer.camera.camera.get_projection_matrix()[:3, :3]
draw_pipeline = RGBMaskToPose6D(model, object_sizes, camera, draw=True)
draw = DrawInferences(experiment_path, images, draw_pipeline)
callbacks = [log, stop, save, plateau, draw]

# saving hyper-parameters and model summary
with open(os.path.join(experiment_path, 'hyperparameters.json'), 'w') as filer:
    json.dump(args.__dict__, filer, indent=4)
with open(os.path.join(experiment_path, 'model_summary.txt'), 'w') as filer:
    model.summary(print_fn=lambda x: filer.write(x + '\n'))

model.fit(
    sequence,
    epochs=args.max_num_epochs,
    callbacks=callbacks,
    verbose=1,
    workers=0)
