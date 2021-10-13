import os
import glob
import json
import argparse
import numpy as np
import time
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import tensorflow as tf

import neptune

from paz.backend.image import write_image
from paz.abstract import GeneratingSequence
from paz.abstract.sequence import GeneratingSequence
from paz.optimization.callbacks import DrawInferences
from paz.pipelines import AutoEncoderPredictor

from scenes import SingleView

from pipelines import GeneratedVectorGenerator
from model import Pecors, pecors_loss, PlotImagesCallback, NeptuneLogger


root_path = os.path.join(os.path.expanduser('~'), '.keras/')
parser = argparse.ArgumentParser()
parser.add_argument('-id', '--background_images_directory', type=str,
                    help='Path to directory containing background images',
                    default="/media/fabian/Data/Masterarbeit/data/VOCdevkit/VOC2012/JPEGImages")
parser.add_argument('-pi', '--images_directory', type=str,
                    help='Path to pre-generated images (npy format)',
                    default="/media/fabian/Data/Masterarbeit/data/VOCdevkit/VOC2012/JPEGImages")
parser.add_argument('-bs', '--batch_size', default=4, type=int,
                    help='Batch size for training')
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float,
                    help='Initial learning rate for Adam')
parser.add_argument('-ld', '--image_size', default=128, type=int,
                    help='Size of the side of a square image e.g. 64')
parser.add_argument('-sp', '--stop_patience', default=7, type=int,
                    help='Number of epochs before doing early stopping')
parser.add_argument('-pp', '--plateau_patience', default=3, type=int,
                    help='Number of epochs before reducing learning rate')
parser.add_argument('-e', '--max_num_epochs', default=10000, type=int,
                    help='Maximum number of epochs before finishing')
parser.add_argument('-st', '--steps_per_epoch', default=5, type=int,
                    help='Steps per epoch')
parser.add_argument('-sh', '--top_only', default=0, choices=[0, 1], type=int,
                    help='Flag for full sphere or top half for rendering')
parser.add_argument('-ls', '--loss', type=str,
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
parser.add_argument('-nc', '--neptune_config',
                    type=str, help='Path to config file where Neptune Token and project name is stored')
parser.add_argument('-ni', '--neptune_log_interval',
                    type=int, default=100, help='How long (in epochs) to wait for the next Neptune logging')
parser.add_argument('-rm', '--rotation_matrices',
                    type=str, help='Path to npy file with a list of rotation matrices')
parser.add_argument('-de', '--description',
                    type=str, help='Description of the model')
args = parser.parse_args()


pecors = Pecors()

optimizer = Adam(args.learning_rate, amsgrad=True)
pecors.compile(optimizer=optimizer, loss=pecors_loss, run_eagerly=True)
pecors.summary()


# creating sequencer
background_image_paths = glob.glob(os.path.join(args.background_images_directory, '*.jpg'))

processor_train = GeneratedVectorGenerator(os.path.join(args.images_directory, "test"), background_image_paths, image_size=args.image_size, num_occlusions=0)
processor_test = GeneratedVectorGenerator(os.path.join(args.images_directory, "test"), background_image_paths, image_size=args.image_size, num_occlusions=0)
sequence_train = GeneratingSequence(processor_train, args.batch_size, args.steps_per_epoch)
sequence_test = GeneratingSequence(processor_test, args.batch_size, args.steps_per_epoch)

# set up neptune run
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
       upload_stdout=False,
       description=args.description,
       params={'batch_size': args.batch_size, 'learning_rate': args.learning_rate, 'steps_per_epoch': args.steps_per_epoch}
    )

    neptune_callback = NeptuneLogger(pecors, log_interval=args.neptune_log_interval, save_path=args.save_path)
    callbacks.append(neptune_callback)


pecors.fit_generator(
    sequence_train,
    steps_per_epoch=args.steps_per_epoch,
    epochs=args.max_num_epochs,
    callbacks=callbacks,
    verbose=1,
    workers=0)