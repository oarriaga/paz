import os
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['PYOPENGL_PLATFORM'] = 'egl'

import glob
import logging
import argparse

import numpy as np
from paz.models import Projector
from paz.pipelines import KeypointAugmentation
from paz.core.sequencer import GeneratingSequencer
from paz.core import ops as p

from poseur.scenes import SingleView


keypoints_path='/home/octavio/.keras/paz/models/keypointnet-shared_10_035_power_drill/keypoints_mean.txt'
# keypoints_path='/home/octavio/Downloads/keypoints_mean.txt'

description = 'Training script for learning 2D probabilistic keypoints'
parser = argparse.ArgumentParser(description=description)
parser.add_argument(
    '-op', '--obj_path',
    default=os.path.join(
        os.path.expanduser('~'),
        '.keras/paz/datasets/ycb/models/035_power_drill/textured.obj'),
    type=str, help='Path for writing model weights and logs')
parser.add_argument(
    '-id', '--images_directory',
    default=os.path.join(
        os.path.expanduser('~'),
        '.keras/paz/datasets/voc-backgrounds/'),
    type=str, help='Path to directory containing background images')
parser.add_argument('-kp', '--keypoints_path', default=None, type=str,
                    help='Fullpath to file containing model 3D keypoints')
parser.add_argument('-bs', '--batch_size', default=36, type=int,
                    help='Batch size for training')
parser.add_argument('-is', '--image_size', default=128, type=int,
                    help='Size of the side of a square image e.g. 64')
parser.add_argument('-sh', '--sphere', default='full',
                    choices=['full', 'half'], type=str,
                    help='Flag for full sphere or top half for rendering')
parser.add_argument('-r', '--roll', default=3.14159, type=float,
                    help='Threshold for camera roll in radians')
parser.add_argument('-t', '--translation', default=0.05, type=float,
                    help='Threshold for translation')
parser.add_argument('-d', '--depth', nargs='+', type=float,
                    default=[0.3, 0.5],
                    help='Distance from camera to origin in meters')
parser.add_argument('-l', '--light', nargs='+', type=float,
                    default=[0.3, 30.0],
                    help='Light intensity from poseur')
parser.add_argument('-s', '--shift', default=0.05, type=float,
                    help='Threshold of random shift of camera')
parser.add_argument('-fv', '--y_fov', default=3.14159 / 4.0, type=float,
                    help='Field of view angle in radians')
parser.add_argument('-oc', '--num_occlusions', default=2, type=int,
                    help='Number of occlusions')
args = parser.parse_args()


# loading keypoints
keypoints = np.loadtxt(keypoints_path)
num_keypoints = len(keypoints)

if args.images_directory is None:
    logging.warning('Image path was not given not given.'
                    'Augmentations will be ran with a plain background color')
    image_paths = None
else:
    image_paths = glob.glob(os.path.join(args.images_directory, '*.png'))

if len(image_paths) == 0:
    raise ValueError('PNG files were not found in', args.images_directory)

# setting scene
renderer = SingleView(
    args.obj_path,
    (args.image_size, args.image_size),
    args.y_fov,
    args.depth,
    args.sphere,
    args.roll,
    args.translation,
    args.shift,
    args.light)

focal_length = renderer.camera.get_projection_matrix()[0, 0]
projector = Projector(focal_length, use_numpy=True)
processor = KeypointAugmentation(
    renderer,
    projector,
    keypoints,
    'train',
    image_paths,
    args.image_size,
    False,
    args.num_occlusions)


# creating sequencer
sequencer = GeneratingSequencer(processor, args.batch_size)
images = (sequencer.__getitem__(0)[0]['image'] * 255).astype('uint8')
mosaic = p.make_mosaic(images, (6, 6))
p.show_image(mosaic.astype('uint8'))
