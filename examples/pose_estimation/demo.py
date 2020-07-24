import os
import argparse
import numpy as np

from tensorflow.keras.utils import get_file

from paz.backend.camera import Camera
from paz.backend.camera import VideoPlayer
from paz.models import HaarCascadeDetector
from paz.models import KeypointNet2D

from pipelines import HeadPose6DEstimation
from pipelines import model_data

description = 'Demo script for estimating 6D pose-heads from face-keypoints'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-f', '--filters', default=32, type=int,
                    help='Number of filters in convolutional blocks')
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
parser.add_argument('-d', '--detector_name', type=str,
                    default='frontalface_default')
parser.add_argument('-nk', '--num_keypoints', default=15, type=int,
                    help='Number of keypoints')
parser.add_argument('-is', '--image_size', default=96, type=int,
                    help='Model image size')
parser.add_argument('-fl', '--focal_length', type=float, default=None,
                    help="Focal length in pixels. If ''None'' it's"
                    "approximated using the image width")
parser.add_argument('-ic', '--image_center', nargs='+', type=float,
                    default=None, help="Image center in pixels for internal"
                    "camera matrix. If ''None'' it's approximated using the"
                    "image center from an extracted frame.")
parser.add_argument('-wu', '--weights_URL', type=str,
                    default='https://github.com/oarriaga/altamira-data/'
                    'releases/download/v0.7/', help='URL to keypoint weights')
# https://github.com/oarriaga/altamira-data/releases/download/v0.7/FaceKP_keypointnet2D_32_15_weights.hdf5
args = parser.parse_args()

# obtaining a frame to perform focal-length and camera center approximation
camera = Camera(args.camera_id)
camera.start()
image_size = camera.read().shape[0:2]
camera.stop()

# loading focal length or approximating it
focal_length = args.focal_length
if focal_length is None:
    focal_length = image_size[1]

# loading image/sensor center or approximating it
image_center = args.image_center
if args.image_center is None:
    image_center = (image_size[1] / 2.0, image_size[0] / 2.0)

# building camera parameters
camera.distortion = np.zeros((4, 1))
camera.intrinsics = np.array([[focal_length, 0, image_center[0]],
                              [0, focal_length, image_center[1]],
                              [0, 0, 1]])

# instantiate model
input_shape = (args.image_size, args.image_size, 1)
model = KeypointNet2D(input_shape, args.num_keypoints, args.filters)
model.summary()

# loading weights
model_name = ['FaceKP', model.name, str(args.filters), str(args.num_keypoints)]
model_name = '%s_weights.hdf5' % '_'.join(model_name)
URL = args.weights_URL + model_name
weights_path = get_file(model_name, URL, cache_subdir='paz/models')
model.load_weights(weights_path)
model.compile(run_eagerly=False)

# setting detector
detector = HaarCascadeDetector(args.detector_name, 0)

# setting prediction pipeline
pipeline = HeadPose6DEstimation(detector, model, model_data, camera)

# setting camera and video player
player = VideoPlayer((640, 480), pipeline, camera)
player.run()
