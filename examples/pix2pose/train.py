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
from paz.abstract.sequence import GeneratingSequencePix2Pose, GeneratingSequence
from paz.optimization.callbacks import DrawInferences
from paz.pipelines import AutoEncoderPredictor

from scenes import SingleView

from pipelines import DepthImageGenerator, RendererDataGenerator, make_batch_discriminator
from model import Generator, Discriminator, transformer_loss, loss_color_wrapped, loss_error, PlotImagesCallback, NeptuneLogger


description = 'Training script for learning implicit orientation vector'
root_path = os.path.join(os.path.expanduser('~'), '.keras/')
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-op', '--obj_path', type=str, help='Path of 3D OBJ model',
                    default=os.path.join(
                        root_path,
                        'datasets/035_power_drill/tsdf/textured.obj'))
parser.add_argument('-cl', '--class_name', default='035_power_drill', type=str,
                    help='Class name to be added to model save path')
parser.add_argument('-id', '--images_directory', type=str,
                    help='Path to directory containing background images',
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
parser.add_argument('-ls', '--loss', default=transformer_loss, type=str,
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
args = parser.parse_args()

# setting optimizer and compiling model
dcgan_input = Input(shape=(128, 128, 3))
discriminator = Discriminator()
generator = Generator()
color_output, error_output = generator(dcgan_input)
discriminator.trainable = False
discriminator_output = discriminator(color_output)
dcgan = Model(inputs=[dcgan_input], outputs={"color_output": color_output, "error_output": error_output, "discriminator_output": discriminator_output})

rotation_matrices = np.load(args.rotation_matrices)
loss_color = loss_color_wrapped(rotation_matrices)

optimizer = Adam(args.learning_rate, amsgrad=True)
losses = {"color_output": loss_color,
          "error_output": loss_error,
          "discriminator_output": "binary_crossentropy"}
lossWeights = {"color_output": 100.0, "error_output": 50.0, "discriminator_output": 1.0}
dcgan.compile(optimizer=optimizer, loss=losses, loss_weights=lossWeights)
dcgan.summary()

discriminator.trainable = True
discriminator.compile(loss=['binary_crossentropy'], optimizer=optimizer)
discriminator.summary()

# setting scene
renderer = SingleView(args.obj_path, (args.image_size, args.image_size),
                      args.y_fov, args.depth, args.light, bool(args.top_only),
                      args.roll, args.shift)

# creating sequencer
image_paths = glob.glob(os.path.join(args.images_directory, '*.jpg'))
processor = DepthImageGenerator(renderer, args.image_size, image_paths, num_occlusions=0)
sequence = GeneratingSequencePix2Pose(processor, dcgan, args.batch_size, args.steps_per_epoch*2, rotation_matrices=rotation_matrices)

# making directory for saving model weights and logs
model_name = '_'.join([dcgan.name, args.class_name])
save_path = os.path.join(args.save_path, model_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)

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
       upload_source_files=["train.py", "scenes.py", "predict.py", "pipelines.py", "model.py"],
       description='VOC backgrounds, complete GAN architecture, weighted losses',
       params={'batch_size': args.batch_size, 'learning_rate': args.learning_rate, 'steps_per_epoch': args.steps_per_epoch}
    )

# setting callbacks
log = CSVLogger(os.path.join(save_path, '%s.log' % model_name))
log.model = dcgan
stop = EarlyStopping('loss', patience=args.stop_patience, verbose=1)
stop.model = dcgan
plateau = ReduceLROnPlateau('loss', patience=args.plateau_patience, verbose=1)
plateau.model = dcgan
model_path = os.path.join(save_path, '%s_weights.hdf5' % model_name)
save = ModelCheckpoint(
    model_path, 'loss', verbose=1, save_best_only=True, save_weights_only=False)
save.model = dcgan

plot = PlotImagesCallback(dcgan, sequence, save_path, args.obj_path, args.image_size, args.y_fov, args.depth, args.light,
                 args.top_only, args.roll, args.shift, args.images_directory, args.batch_size, args.steps_per_epoch, args.neptune_config is not None, rotation_matrices=rotation_matrices)

callbacks=[stop, log, save, plateau, plot]

if args.neptune_config is not None:
    neptune_callback = NeptuneLogger(dcgan, log_interval=args.neptune_log_interval)
    callbacks.append(neptune_callback)

# saving hyper-parameters and model summary as text files
args.__dict__['loss'] = args.__dict__['loss'].__name__
print(save_path)
print(args.__dict__)
with open(os.path.join(save_path, 'hyperparameters.json'), 'w') as filer:
    json.dump(args.__dict__, filer, indent=4)
with open(os.path.join(save_path, 'model_summary.txt'), 'w') as filer:
    dcgan.summary(print_fn=lambda x: filer.write(x + '\n'))


for callback in callbacks:
    callback.on_train_begin()

for num_epoch in range(args.max_num_epochs):
    sequence_iterator = sequence.__iter__()
    for callback in callbacks:
        callback.on_epoch_begin(num_epoch)

    for num_batch in range(args.steps_per_epoch):
        # Train the discriminator
        discriminator.trainable = True
        start = time.time()
        batch = next(sequence_iterator)
        end = time.time()
        print("Batch generation time: {}".format(end - start))

        start = time.time()

        X_discriminator_real, y_discriminator_real = make_batch_discriminator(generator, batch[0]['input_image'], batch[1]['color_output'], 1)
        loss_discriminator_real = discriminator.train_on_batch(X_discriminator_real, y_discriminator_real)

        X_discriminator_fake, y_discriminator_fake = make_batch_discriminator(generator, batch[0]['input_image'], batch[1]['color_output'], 0)
        loss_discriminator_fake = discriminator.train_on_batch(X_discriminator_fake, y_discriminator_fake)
        end = time.time()
        print("Train time discriminator: {}".format(end - start))
        loss_discriminator = (loss_discriminator_real + loss_discriminator_fake)/2.

        # Train the generator
        discriminator.trainable = False

        start = time.time()
        loss_dcgan, loss_color_output, loss_dcgan_discriminator, loss_error_output = dcgan.train_on_batch(batch[0]['input_image'], {"color_output": batch[1]['color_output'], "error_output": batch[1]['error_output'], "discriminator_output": np.ones((args.batch_size, 1))})

        # Test the network
        batch_test = next(sequence_iterator)
        loss_dcgan_test, loss_color_output_test, loss_dcgan_discriminator_test, loss_error_output_test = dcgan.test_on_batch(batch_test[0]['input_image'], {"color_output": batch_test[1]['color_output'], "error_output": batch_test[1]['error_output'], "discriminator_output": np.ones((args.batch_size, 1))})

        end = time.time()
        print("Train time DCGAN: {}".format(end - start))

        print("Loss discriminator: {}, Loss DCGAN: {}, Loss DCGAN color output: {}, Loss DCGAN discriminator: {}, Loss error output: {}".format(loss_discriminator, loss_dcgan, loss_color_output, loss_dcgan_discriminator, loss_error_output))
        print("--------------------------------------")

    for callback in callbacks:
        callback.on_epoch_end(num_epoch, logs={'loss_discriminator': loss_discriminator,
                                               'loss_dcgan': loss_dcgan, 'loss_color_output': loss_color_output,
                                               'loss_dcgan_discriminator': loss_dcgan_discriminator,
                                               'loss_error_output': loss_error_output,
                                               'loss_dcgan_test': loss_dcgan_test, 'loss_color_output_test': loss_color_output_test,
                                               'loss_dcgan_discriminator_test': loss_dcgan_discriminator_test,
                                               'loss_error_output_test': loss_error_output_test
                                               })


for callback in callbacks:
    callback.on_train_end()

if args.neptune_config is not None:
    neptune.stop()