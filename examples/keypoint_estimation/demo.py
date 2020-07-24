import argparse

from paz.backend.camera import Camera
from paz.backend.camera import VideoPlayer
from paz.models import HaarCascadeDetector
from paz.models import KeypointNet2D

from pipelines import PredictMultipleKeypoints2D
from tensorflow.keras.utils import get_file

description = 'Demo script for running 2D probabilistic keypoints'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-f', '--filters', default=32, type=int,
                    help='Number of filters in convolutional blocks')
parser.add_argument('-is', '--image_size', default=96, type=int,
                    help='Model image size')
parser.add_argument('-nk', '--num_keypoints', default=15, type=int,
                    help='Number of keypoints')
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
parser.add_argument('-d', '--detector_name', type=str,
                    default='frontalface_default')
parser.add_argument('-wp', '--weights_path', default=None,
                    type=str, help='Path of trained model weights')
parser.add_argument('-wu', '--weights_URL', type=str,
                    default='https://github.com/oarriaga/altamira-data/'
                    'releases/download/v0.7/', help='URL to keypoint weights')
args = parser.parse_args()

weights_path = args.weights_path
# instantiating model
if weights_path is None:
    model = KeypointNet2D((96, 96, 1), 15, 32)
else:
    input_shape = (args.image_size, args.image_size, 1)
    model = KeypointNet2D(input_shape, args.num_keypoints, args.filters)
model.summary()

# loading weights
if weights_path is None:
    model_name = '_'.join(['FaceKP', model.name, '32', '15'])
    model_name = '%s_weights.hdf5' % model_name
    URL = args.weights_URL + model_name
    print(URL)
    weights_path = get_file(model_name, URL, cache_subdir='paz/models')
model.load_weights(weights_path)


# setting detector
detector = HaarCascadeDetector(args.detector_name, 0)

# setting prediction pipeline
pipeline = PredictMultipleKeypoints2D(detector, model)

# setting camera and video player
camera = Camera(args.camera_id)
player = VideoPlayer((640, 480), pipeline, camera)
player.run()
