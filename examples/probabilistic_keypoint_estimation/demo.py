import os
import argparse
import numpy as np

from paz.backend.camera import Camera
from paz.backend.camera import VideoPlayer
from paz.models import HaarCascadeDetector

from model import GaussianMixtureModel
from pipelines import ProbabilisticKeypointPrediction


description = 'Demo script for running 2D probabilistic keypoints'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-f', '--filters', default=8, type=int,
                    help='Number of filters in convolutional blocks')
parser.add_argument('-nk', '--num_keypoints', default=15, type=int,
                    help='Number of keypoints')
parser.add_argument('-is', '--image_size', default=96, type=int,
                    help='Image size')
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
batch_shape = (1, args.image_size, args.image_size, 1)
model = GaussianMixtureModel(batch_shape, args.num_keypoints, args.filters)
model.summary()

# loading weights
model_name = ['FaceKP', model.name, str(args.filters), str(args.num_keypoints)]
model_name = '_'.join(model_name)
save_path = os.path.join(args.save_path, model_name)
model_path = os.path.join(save_path, '%s_weights.hdf5' % model_name)
model.load_weights(model_path)
model.compile(run_eagerly=False)

model.predict(np.zeros((1, 96, 96, 1)))  # first prediction takes a while...
# setting detector
detector = HaarCascadeDetector(args.detector_name, 0)

# setting prediction pipeline
pipeline = ProbabilisticKeypointPrediction(detector, model)

# setting camera and video player
camera = Camera(args.camera_id)
player = VideoPlayer((640, 480), pipeline, camera)
player.run()
