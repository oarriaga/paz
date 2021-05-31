import os
import glob
import json
import argparse

import numpy as np
import neptune

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from paz.backend.image import write_image
from paz.abstract import GeneratingSequence
from paz.optimization.callbacks import DrawInferences
from paz.pipelines import AutoEncoderPredictor

from scenes import SingleView

from pipelines import ImageGenerator
from model import DOPE, NeptuneLogger, PlotImagesCallback

description = 'Training script for learning implicit orientation vector'
root_path = os.path.join(os.path.expanduser('~'), '.keras/paz/')
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-op', '--obj_path', nargs='+', type=str, help='Paths of 3D OBJ models',
                    default=os.path.join(
                        root_path,
                        'datasets/ycb/models/035_power_drill/textured.obj'))
parser.add_argument('-cl', '--class_name', default='035_power_drill', type=str,
                    help='Class name to be added to model save path')
parser.add_argument('-id', '--images_directory', type=str,
                    help='Path to directory containing background images',
                    default=None)
parser.add_argument('-bs', '--batch_size', default=2, type=int,
                    help='Batch size for training')
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float,
                    help='Initial learning rate for Adam')
parser.add_argument('-is', '--latent_dimension', default=128, type=int,
                    help='Latent dimension of the auto-encoder')
parser.add_argument('-ld', '--image_size', default=128, type=int,
                    help='Size of the side of a square image e.g. 64')
parser.add_argument('-sp', '--stop_patience', default=7, type=int,
                    help='Number of epochs before doing early stopping')
parser.add_argument('-pp', '--plateau_patience', default=3, type=int,
                    help='Number of epochs before reducing learning rate')
parser.add_argument('-e', '--max_num_epochs', default=10000, type=int,
                    help='Maximum number of epochs before finishing')
parser.add_argument('-st', '--steps_per_epoch', default=1000, type=int,
                    help='Steps per epoch')
parser.add_argument('-sh', '--top_only', default=0, choices=[0, 1], type=int,
                    help='Flag for full sphere or top half for rendering')
parser.add_argument('-ls', '--loss', default='binary_crossentropy', type=str,
                    help='tf.keras loss function name to be used')
parser.add_argument('-r', '--roll', default=3.14159, type=float,
                    help='Threshold for camera roll in radians')
parser.add_argument('-s', '--shift', default=0.05, type=float,
                    help='Threshold of random shift of camera')
parser.add_argument('-d', '--depth', nargs='+', type=float,
                    default=[0.3, 0.5],
                    help='Distance from camera to origin in meters')
parser.add_argument('-fv', '--y_fov', default=3.14159 / 4.0, type=float,
                    help='Field of view angle in radians')
parser.add_argument('-l', '--light', nargs='+', type=float,
                    default=[.5, 30],
                    help='Light intensity from poseur')
parser.add_argument('-oc', '--num_occlusions', default=2, type=int,
                    help='Number of occlusions')
parser.add_argument('-sa', '--save_path',
                    default=os.path.join(
                        os.path.expanduser('~'), '.keras/paz/models'),
                    type=str, help='Path for writing model weights and logs')
parser.add_argument('-sf', '--scaling_factor', default=8.0, type=float,
                    help='Downscaling factor of the images')
parser.add_argument('-ns', '--num_stages', default=3, type=int,
                    help='Number of stages for DOPE')
parser.add_argument('-nc', '--neptune_config',
                    type=str, help='Path to config file where Neptune Token and project name is stored')
parser.add_argument('-ug', '--use_generator', default=1, choices=[0, 1], type=int,
                    help='Use generator to generate data or use already generated data')
parser.add_argument('-pd', '--path_data', type=str, help='Path for the training data')

args = parser.parse_args()


# setting optimizer and compiling model
latent_dimension = args.latent_dimension
model = DOPE(num_stages=args.num_stages, image_shape=(args.image_size, args.image_size, 3))
optimizer = Adam(args.learning_rate, amsgrad=True)

# Add losses for all the stages
losses = dict()
for i in range(1, args.num_stages+1):
    losses['belief_maps_stage_' + str(i)] = 'mse'
    #losses['affinity_maps_stage_' + str(i)] = 'mse'

print(losses)
model.compile(optimizer, losses, metrics=['mse'])
model.summary()

# setting scene
print(args.obj_path)
colors = [np.array([255, 0, 0]), np.array([0, 255, 0])]
renderer = SingleView(filepath=args.obj_path, colors=colors, viewport_size=(args.image_size, args.image_size),
                      y_fov=args.y_fov, distance=args.depth, light_bounds=args.light, top_only=bool(args.top_only),
                      roll=args.roll, shift=args.shift)

# creating sequencer
if not (args.images_directory is None):
    image_paths = glob.glob(os.path.join(args.images_directory, '*.png'))
else:
    image_paths = None

processor = ImageGenerator(renderer, args.image_size, int(args.image_size/args.scaling_factor), image_paths, args.num_occlusions, num_stages=3)

sequence = GeneratingSequence(processor, args.batch_size, args.steps_per_epoch)

# making directory for saving model weights and logs
model_name = '_'.join([model.name, str(latent_dimension), args.class_name])
save_path = os.path.join(args.save_path, model_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# setting callbacks
log = CSVLogger(os.path.join(save_path, '%s.log' % model_name))
stop = EarlyStopping('loss', patience=args.stop_patience, verbose=1)
plateau = ReduceLROnPlateau('loss', patience=args.plateau_patience, verbose=1)
model_path = os.path.join(save_path, '%s_weights.hdf5' % model_name)
save = ModelCheckpoint(
    model_path, 'loss', verbose=1, save_best_only=True, save_weights_only=True)

# setting drawing callbacks
"""
images = (sequence.__getitem__(0)[0]['input_1'] * 255).astype('uint8')
for arg, image in enumerate(images):
    image_name = 'image_%03d.png' % arg
    image_path = os.path.join(save_path, 'original_images/' + image_name)
    write_image(image_path, image)
inferencer = AutoEncoderPredictor(model)
draw = DrawInferences(save_path, images, inferencer)
"""

# saving hyper-parameters and model summary as text files
print(save_path)
with open(os.path.join(save_path, 'hyperparameters.json'), 'w') as filer:
    json.dump(args.__dict__, filer, indent=4)
with open(os.path.join(save_path, 'model_summary.txt'), 'w') as filer:
    model.summary(print_fn=lambda x: filer.write(x + '\n'))

callbacks=[stop, log, save, plateau]

# Set up neptune run
if args.neptune_config is not None:
    neptune_config_file = open(args.neptune_config)
    neptune_config = neptune_config_file.read().split('\n')
    neptune_token = neptune_config[0]
    neptune_experiment_name = neptune_config[1]
    neptune_run_name = neptune_config[2]

    neptune.init(
       api_token=neptune_token,
       project_qualified_name=neptune_experiment_name
    )

    neptune.create_experiment(
       name=neptune_run_name,
       description='VOC backgrounds',
       params={'batch_size': args.batch_size, 'learning_rate': args.learning_rate, 'steps_per_epoch': args.steps_per_epoch}
    )

    neptuneLogger = NeptuneLogger(model)
    callbacks.append(neptuneLogger)

plotCallback = PlotImagesCallback(model, sequence, neptune_logging=(args.neptune_config is not None), num_stages=args.num_stages)
callbacks.append(plotCallback)

# model optimization
if bool(args.use_generator):
    model.fit_generator(
        sequence,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.max_num_epochs,
        callbacks=callbacks,
        verbose=1,
        workers=0)
else:
    images = np.load(os.path.join(args.path_data, "images_batch_1.npy"))
    belief_maps = np.load(os.path.join(args.path_data, "belief_maps_batch_1.npy"))

    # Normalize images
    images = images.astype(np.float32)/255.

    model.fit(
        x=[images],
        y=[belief_maps, belief_maps, belief_maps],
        batch_size=args.batch_size,
        epochs=args.max_num_epochs,
        callbacks=callbacks,
        verbose=1,
        workers=0)
