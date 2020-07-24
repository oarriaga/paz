import os
import argparse
import numpy as np

from paz.backend.camera import Camera
from paz.backend.camera import VideoPlayer
from paz.models import HaarCascadeDetector
from paz.models import KeypointNet2D

from pipelines import HeadPose6DEstimation
# TODO change camera id to 0
description = 'Demo script for running 2D probabilistic keypoints'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-f', '--filters', default=32, type=int,
                    help='Number of filters in convolutional blocks')
parser.add_argument('-nk', '--num_keypoints', default=15, type=int,
                    help='Number of keypoints')
parser.add_argument('-is', '--image_size', default=96, type=int,
                    help='Model image size')
parser.add_argument('-c', '--camera_id', type=int, default=2,
                    help='Camera device ID')
parser.add_argument('-d', '--detector_name', type=str,
                    default='frontalface_default')
parser.add_argument('-s', '--save_path',
                    default=os.path.join(
                        os.path.expanduser('~'), '.keras/paz/models'),
                    type=str, help='Path for writing model weights and logs')
args = parser.parse_args()

keypoints3D = np.array([
    [-220, 678, 1138],  # left--center-eye
    [+220, 678, 1138],  # right-center-eye
    [-131, 676, 1107],  # left--eye close to nose
    [-294, 610, 1123],  # left--eye close to ear
    [+131, 676, 1107],  # right-eye close to nose
    [+294, 610, 1123],  # right-eye close to ear
    [-106, 758, 1224],  # left--eyebrow close to nose
    [-375, 585, 1208],  # left--eyebrow close to ear
    [+106, 758, 1224],  # right-eyebrow close to nose
    [+375, 585, 1208],  # right-eyebrow close to ear
    [0.0, 909, 919],  # nose
    [-183, 691, 683],  # lefty-lip
    [+183, 691, 683],  # right-lip
    [0.0, 826, 754],  # up---lip
    [0.0, 815, 645],  # down-lip
])

keypoints3D = keypoints3D - np.mean(keypoints3D, axis=0)
model_points = {'keypoints3D': keypoints3D,
                'dimensions': {None: [900.0, 600.0]}}

camera = Camera(args.camera_id)
camera.start()
frame = camera.read()
camera.stop()

image_size = frame.shape[0:2]
focal_length = image_size[1]
#  center = (image_size[1] / 2.0, image_size[0] / 2.0)
center = (300, 200)
focal_length = 832.0
camera.distortion = np.zeros((4, 1))
camera.intrinsics = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]])

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

# setting detector
detector = HaarCascadeDetector(args.detector_name, 0)

# setting prediction pipeline
pipeline = HeadPose6DEstimation(detector, model, model_points, camera)

# setting camera and video player
player = VideoPlayer((640, 480), pipeline, camera)
player.run()
