import argparse
import os
import glob
import numpy as np
import trimesh

from paz.abstract import GeneratingSequence
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam

from model import Deeplabv3, epos_loss_wrapped
from pipelines import GeneratedImageGenerator
from callbacks import NeptuneCallback, PlotImagesCallback

parser = argparse.ArgumentParser()
parser.add_argument('-id', '--background_images_directory', type=str,
                    help='Path to directory containing background images',
                    default="/media/fabian/Data/Masterarbeit/data/VOCdevkit/VOC2012/JPEGImages")
parser.add_argument('-pi', '--images_directory', type=str,
                    help='Path to pre-generated images (npy format)',
                    default="/media/fabian/Data/Masterarbeit/data/VOCdevkit/VOC2012/JPEGImages")
parser.add_argument('-bs', '--batch_size', default=4, type=int,
                    help='Batch size for training')
parser.add_argument('-af', '--activation_function', default="sigmoid", type=str,
                    help='Name activation funcion')
parser.add_argument('-o', '--num_objects', default=1, type=int,
                    help='Number of objects this net is trained on')
parser.add_argument('-nf', '--num_fragments', default=5, type=int,
                    help='Number of fragments')
parser.add_argument('-op', '--obj_path', type=str, help='Path to the object obj file')
parser.add_argument('-fc', '--fragment_center_path', type=str, help='Path to the object obj file')

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
parser.add_argument('-no', '--num_occlusions', default=2, type=int,
                    help='Number of occlusions')
parser.add_argument('-sa', '--save_path',
                    default=os.path.join(
                        os.path.expanduser('~'), '.keras/paz/models'),
                    type=str, help='Path for writing model weights and logs')
parser.add_argument('-nc', '--neptune_config',
                    type=str, help='Path to config file where Neptune Token and project name is stored')
parser.add_argument('-ni', '--neptune_log_interval',
                    type=int, default=100, help='How long (in epochs) to wait for the next Neptune logging')
parser.add_argument('-de', '--description',
                    type=str, help='Description of the model')
args = parser.parse_args()

num_output_channels = (4*args.num_objects*args.num_fragments + args.num_objects + 1)
deeplabv3 = Deeplabv3(weights=None, input_tensor=None, input_shape=(args.image_size, args.image_size, 3),
                      num_objects=args.num_objects, num_fragments=args.num_fragments, backbone='mobilenetv2',
                      OS=16, alpha=1., activation=args.activation_function)
deeplabv3.summary()
plot_model(deeplabv3, "deeplabv3.png")

background_image_paths = glob.glob(os.path.join(args.background_images_directory, '*.jpg'))
processor_train = GeneratedImageGenerator(os.path.join(args.images_directory, "train"), background_image_paths, args.image_size, num_output_channels)
processor_test = GeneratedImageGenerator(os.path.join(args.images_directory, "test"), background_image_paths, args.image_size, num_output_channels)
sequence_train = GeneratingSequence(processor_train, args.batch_size, args.steps_per_epoch)
sequence_test = GeneratingSequence(processor_test, args.batch_size, args.steps_per_epoch)

epos_loss = epos_loss_wrapped(args.num_objects, args.num_fragments)
optimizer = Adam(args.learning_rate, amsgrad=True)
deeplabv3.compile(optimizer, epos_loss, metrics=['mse'])#, run_eagerly=True)

callbacks = list()

if not args.neptune_config is None:
    neptune_callback = NeptuneCallback(model=deeplabv3, save_path=args.save_path)
    callbacks.append(neptune_callback)

loaded_trimesh = trimesh.load(args.obj_path)
fragment_centers = np.load(args.fragment_center_path)
plot_callback = PlotImagesCallback(deeplabv3, sequence_test, args.num_objects, args.num_fragments, fragment_centers, loaded_trimesh.extents)
callbacks.append(plot_callback)

# Train the model
deeplabv3.fit_generator(
    sequence_train,
    steps_per_epoch=args.steps_per_epoch,
    epochs=args.max_num_epochs,
    callbacks=callbacks,
    verbose=1,
    workers=0)