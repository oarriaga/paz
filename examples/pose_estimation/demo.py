import os
import argparse
import numpy as np

from paz.backend.camera import Camera
from paz.backend.camera import VideoPlayer
from paz.models import HaarCascadeDetector
from paz.models import KeypointNet2D

from pipelines import PredictMultipleKeypoints2D

description = 'Demo script for running 2D probabilistic keypoints'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-f', '--filters', default=8, type=int,
                    help='Number of filters in convolutional blocks')
parser.add_argument('-nk', '--num_keypoints', default=15, type=int,
                    help='Number of keypoints')
parser.add_argument('-is', '--image_size', default=96, type=int,
                    help='Model image size')
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
parser.add_argument('-d', '--detector_name', type=str,
                    default='frontalface_default')
parser.add_argument('-s', '--save_path',
                    default=os.path.join(
                        os.path.expanduser('~'), '.keras/paz/models'),
                    type=str, help='Path for writing model weights and logs')
args = parser.parse_args()

# instantiate model
input_shape = (args.image_size, args.image_size, 1)
model = KeypointNet2D(input_shape, args.num_keypoints, args.filters)
model.summary()

# loading weights
model_name = ['FaceKP', model.name, str(args.filters), str(args.num_keypoints)]
model_name = '_'.join(model_name)
save_path = os.path.join(args.save_path, model_name)
model_path = os.path.join(save_path, '%s_weights.hdf5' % model_name)
model.load_weights(model_path)
model.compile(run_eagerly=False)


# 3D model points.
model_points = np.array([
    [0.0, 9.09, 9.19],  # nose
    [0.0, 8.26, 7.54],  # up---lip
    [0.0, 8.15, 6.45],  # down-lip
    [+1.83, 6.91, 6.83],  # right-lip
    [-1.83, 6.91, 6.83],  # lefty-lip
    [+2.94, 6.10, 11.23],  # right-eye close to ear
    [-2.94, 6.10, 11.23],  # left--eye close to ear
    [+2.20, 6.78, 11.38],  # right-center-eye
    [-2.20, 6.78, 11.38],  # left--center-eye
    [+1.31, 6.76, 11.07],  # right-eye close to nose
    [-1.31, 6.76, 11.07],  # left--eye close to nose
    [+3.75, 5.85, 12.08],  # right-eyebrow close to ear
    [-3.75, 5.85, 12.08],  # left--eyebrow close to ear
    [+1.06, 7.58, 12.24],  # right-eyebrow close to nose
    [-1.06, 7.58, 12.24],  # left--eyebrow close to nose
])


model_points = np.array([
    [+2.20, 6.78, 11.38],  # right-center-eye
    [-2.20, 6.78, 11.38],  # left--center-eye
    [+1.31, 6.76, 11.07],  # right-eye close to nose
    [+2.94, 6.10, 11.23],  # right-eye close to ear
    [-1.31, 6.76, 11.07],  # left--eye close to nose
    [-2.94, 6.10, 11.23],  # left--eye close to ear
    [+1.06, 7.58, 12.24],  # right-eyebrow close to nose
    [+3.75, 5.85, 12.08],  # right-eyebrow close to ear
    [-1.06, 7.58, 12.24],  # left--eyebrow close to nose
    [-3.75, 5.85, 12.08],  # left--eyebrow close to ear
    [0.0, 9.09, 9.19],  # nose
    [+1.83, 6.91, 6.83],  # right-lip
    [-1.83, 6.91, 6.83],  # lefty-lip
    [0.0, 8.26, 7.54],  # up---lip
    [0.0, 8.15, 6.45],  # down-lip
])




image_size = (640, 480)
focal_length = image_size[1]
center = (image_size[1] / 2.0, image_size[0] / 2.0)
camera_matrix = np.array([[focal_length, 0, center[0]],
                          [0, focal_length, center[1]],
                          [0, 0, 1]])


# setting detector
detector = HaarCascadeDetector(args.detector_name, 0)

# setting prediction pipeline
pipeline = PredictMultipleKeypoints2D(detector, model)

# setting camera and video player
camera = Camera(args.camera_id)
player = VideoPlayer((640, 480), pipeline, camera)
player.run()
