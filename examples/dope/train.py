import os
import glob
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
import neptune

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.losses import Reduction

from paz.backend.image import write_image
from paz.abstract import GeneratingSequence
from paz.optimization.callbacks import DrawInferences
from paz.pipelines import AutoEncoderPredictor
from pipelines import GeneratedImageGenerator

from scenes import SingleView

from pipelines import ImageGenerator
from model import DOPE, NeptuneLogger, PlotImagesCallback, custom_mse
from ambiguity import MultipleHypotheses, MultipleHypothesesCallback

description = 'Training script for learning implicit orientation vector'
root_path = os.path.join(os.path.expanduser('~'), '.keras/paz/')
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-op', '--obj_path', nargs='+', type=str, help='Paths of 3D OBJ models',
                    default=os.path.join(
                        root_path,
                        'datasets/ycb/models/035_power_drill/textured.obj'))
parser.add_argument('-cl', '--class_name', default='035_power_drill', type=str,
                    help='Class name to be added to model save path')
parser.add_argument('-id', '--background_images_directory', type=str,
                    help='Path to directory containing background images',
                    default=None)
parser.add_argument('-pi', '--images_directory', type=str,
                    help='Path to pre-generated images (npy format)',
                    default="/media/fabian/Data/Masterarbeit/data/VOCdevkit/VOC2012/JPEGImages")
parser.add_argument('-bs', '--batch_size', default=8, type=int,
                    help='Batch size for training')
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float,
                    help='Initial learning rate for Adam')
parser.add_argument('-is', '--latent_dimension', default=128, type=int,
                    help='Latent dimension of the auto-encoder')
parser.add_argument('-ld', '--image_size', default=400, type=int,
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
                    default=[0.5, 1.0],
                    help='Distance from camera to origin in meters')
parser.add_argument('-fv', '--y_fov', default=3.14159 / 4.0, type=float,
                    help='Field of view angle in radians')
parser.add_argument('-l', '--light', nargs='+', type=float,
                    default=[0.5, 30],
                    help='Light intensity from poseur')
parser.add_argument('-oc', '--num_occlusions', default=0, type=int,
                    help='Number of occlusions')
parser.add_argument('-sa', '--save_path',
                    default=os.path.join(
                        os.path.expanduser('~'), '.keras/paz/models'),
                    type=str, help='Path for writing model weights and logs')
parser.add_argument('-sf', '--scaling_factor', default=8.0, type=float,
                    help='Downscaling factor of the images')
parser.add_argument('-ns', '--num_stages', default=6, type=int,
                    help='Number of stages for DOPE')
parser.add_argument('-nc', '--neptune_config',
                    type=str, help='Path to config file where Neptune Token and project name is stored')
parser.add_argument('-ni', '--neptune_log_interval',
                    type=int, help='How long (in epochs) to wait for the next Neptune logging')
parser.add_argument('-pd', '--path_data', type=str, help='Path for the training data')

args = parser.parse_args()

multipleHypotheses = MultipleHypotheses(M=5)

# setting optimizer and compiling model
latent_dimension = args.latent_dimension
model = DOPE(num_stages=args.num_stages, image_shape=(args.image_size, args.image_size, 3), num_belief_maps=9, multipleHypotheses=multipleHypotheses)
optimizer = Adam(args.learning_rate, amsgrad=True)

# Add losses for all the stages
losses = dict()
for i in range(1, args.num_stages+1):
    loss_unwrapped = multipleHypotheses.loss_multiple_hypotheses_wrapped(custom_mse, 'belief_maps_stage_' + str(i))
    loss_belief_maps = multipleHypotheses.map_layer_names_to_attributes({'belief_maps_stage_' + str(i): loss_unwrapped})
    losses = {**losses, **loss_belief_maps}

print(losses)
model.compile(optimizer, losses, metrics=None, run_eagerly=True)
model.summary()

model.load_weights("/media/fabian/Data/Masterarbeit/data/models/tless06/dope/multiple_hypotheses/dope_model_epoch_6100_weights.h5")
model.save("/media/fabian/Data/Masterarbeit/data/models/tless06/dope/multiple_hypotheses/dope_model_epoch_6100.h5")

"""
background_image_paths = glob.glob(os.path.join(args.background_images_directory, '*.jpg'))

processor_train = GeneratedImageGenerator(os.path.join(args.images_directory, "test"), background_image_paths, num_occlusions=0, image_size_input=args.image_size, image_size_output=int(args.image_size/args.scaling_factor), num_stages=args.num_stages, multipleHypotheses=multipleHypotheses)
processor_test = GeneratedImageGenerator(os.path.join(args.images_directory, "test"), background_image_paths, num_occlusions=0, image_size_input=args.image_size, image_size_output=int(args.image_size/args.scaling_factor), num_stages=args.num_stages, multipleHypotheses=multipleHypotheses)
sequence_train = GeneratingSequence(processor_train, args.batch_size, args.steps_per_epoch)
sequence_test = GeneratingSequence(processor_test, args.batch_size, args.steps_per_epoch)


callbacks = list()

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
       description='',
       upload_stdout=False,
       params={'batch_size': args.batch_size, 'learning_rate': args.learning_rate, 'steps_per_epoch': args.steps_per_epoch}
    )

    neptuneLogger = NeptuneLogger(model, args.neptune_log_interval, args.save_path)
    callbacks.append(neptuneLogger)

plotCallback = PlotImagesCallback(model, sequence_test, neptune_logging=(args.neptune_config is not None), num_stages=args.num_stages, multipleHypotheses=multipleHypotheses)
callbacks.append(plotCallback)

multipleHypothesesCallback = MultipleHypothesesCallback(multipleHypotheses)
callbacks.append(multipleHypothesesCallback)

# We have to do everything manually because otherwise we cannot hand over the current batch to the callback
for num_epoch in range(args.max_num_epochs):
    sequence_iterator_train = sequence_train.__iter__()
    sequence_iterator_test = sequence_test.__iter__()

    for callback in callbacks:
        callback.on_epoch_begin(num_epoch)

    for num_batch in range(args.steps_per_epoch):

        batch_train = next(sequence_iterator_train)
        batch_test = next(sequence_iterator_test)

        for callback in callbacks:
            predictions = model(batch_train[0]['input_1'], training=True)
            print("Predictions type callback: {}".format(type(predictions)))
            callback.on_train_batch_begin(predictions)

        train_on_batch_return = model.train_on_batch(batch_train[0]['input_1'], batch_train[1])
        loss_dope = train_on_batch_return[0]
        loss_belief_maps = list()
        for i in range(1, len(train_on_batch_return), multipleHypotheses.M):
            stage_loss = np.asarray(train_on_batch_return[i:i+multipleHypotheses.M])
            stage_loss = np.mean(stage_loss[np.nonzero(stage_loss)])
            loss_belief_maps.append(stage_loss)

        test_on_batch_return = model.test_on_batch(batch_test[0]['input_1'], batch_test[1])
        loss_dope_test = test_on_batch_return[0]
        loss_belief_maps_test = list()
        for i in range(1, len(test_on_batch_return), multipleHypotheses.M):
            stage_loss = np.asarray(test_on_batch_return[i:i+multipleHypotheses.M])
            stage_loss = np.mean(stage_loss[np.nonzero(stage_loss)])
            loss_belief_maps_test.append(stage_loss)

        loss_dict, loss_dict_test = dict(), dict()
        for i, loss_belief_map in enumerate(loss_belief_maps):
            loss_dict['loss_belief_map_stage_{}'.format(i)] = loss_belief_map

        for i, loss_belief_map in enumerate(loss_belief_maps_test):
            loss_dict_test['loss_belief_map_stage_{}_test'.format(i)] = loss_belief_map

    for callback in callbacks:
        callback.on_epoch_end(num_epoch, logs={**{'loss_dope': loss_dope, 'loss_dope_test': loss_dope_test}, **loss_dict, **loss_dict_test})

for callback in callbacks:
    callback.on_train_end()

if args.neptune_config is not None:
    neptune.stop()
"""