import os
import json
import argparse

from tensorflow.keras.utils import get_file
from sklearn.metrics.pairwise import cosine_similarity as measure
from paz.backend.camera import VideoPlayer, Camera

from scenes import DictionaryView

from model import AutoEncoder
from pipelines import ImplicitRotationPredictor


parser = argparse.ArgumentParser(description='Implicit orientation demo')
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
parser.add_argument('-f', '--y_fov', type=float, default=3.14159 / 4.0,
                    help='field of view')
parser.add_argument('-v', '--viewport_size', type=int, default=128,
                    help='Size of rendered images')
parser.add_argument('-d', '--distance', type=float, default=0.3,
                    help='Distance between camera and 3D model')
parser.add_argument('-s', '--shift', type=float, default=0.01,
                    help='Shift')
parser.add_argument('-l', '--light', type=int, default=10,
                    help='Light intensity')
parser.add_argument('-b', '--background', type=int, default=0,
                    help='Plain background color')
parser.add_argument('-r', '--roll', type=float, default=3.14159,
                    help='Maximum roll')
parser.add_argument('-t', '--translate', type=float, default=0.01,
                    help='Maximum translation')
parser.add_argument('-p', '--top_only', type=int, default=0,
                    help='Rendering mode')
parser.add_argument('--theta_steps', type=int, default=10,
                    help='Amount of steps taken in the X-Y plane')
parser.add_argument('--phi_steps', type=int, default=10,
                    help='Amount of steps taken from the Z-axis')
parser.add_argument('--model_name', type=str,
                    default='VanillaAutoencoder128_128_035_power_drill',
                    help='Model directory name without root')
parser.add_argument('--model_path', type=str,
                    default=os.path.join(
                        os.path.expanduser('~'), '.keras/paz/models/'),
                    help='Root directory PAZ trained models')
args = parser.parse_args()


path = os.path.join(args.model_path, args.model_name)
parameters = json.load(open(os.path.join(path, 'hyperparameters.json'), 'r'))

size = parameters['image_size']
latent_dimension = parameters['latent_dimension']
weights_path = os.path.join(path, args.model_name + '_weights.hdf5')

obj_path = get_file('textured.obj', None,
                    cache_subdir='paz/datasets/ycb/models/035_power_drill/')

renderer = DictionaryView(
    obj_path, (args.viewport_size, args.viewport_size), args.y_fov,
    args.distance, bool(args.top_only), args.light, args.theta_steps,
    args.phi_steps)

encoder = AutoEncoder((size, size, 3), latent_dimension, mode='encoder')
encoder.load_weights(weights_path, by_name=True)
decoder = AutoEncoder((size, size, 3), latent_dimension, mode='decoder')
decoder.load_weights(weights_path, by_name=True)
inference = ImplicitRotationPredictor(encoder, decoder, measure, renderer)
player = VideoPlayer((1280, 960), inference, camera=Camera(args.camera_id))
player.run()
