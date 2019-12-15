import os
import numpy as np

from tensorflow.keras.utils import get_file

from paz.core import VideoPlayer
from paz.models import KeypointNet2D
from paz.pipelines import KeypointToPoseInference
from paz.core.ops import Camera

num_keypoints, class_name, input_shape = 10, '035_power_drill', (128, 128, 3)

# loading points3D
model_name = '_'.join(['keypointnet-shared', str(num_keypoints), class_name])
filename = os.path.join(model_name, 'keypoints_mean.txt')
filename = get_file(filename, None, cache_subdir='paz/models')
points3D = np.loadtxt(filename)[:, :3].astype(np.float64)
points3D = points3D / 10

# loading camera
filename = os.path.join('logitech_c270', 'camera_intrinsics.txt')
filename = get_file(filename, None, cache_subdir='paz/cameras')
intrinsics = np.loadtxt(filename)
filename = os.path.join('logitech_c270', 'distortions.txt')
filename = get_file(filename, None, cache_subdir='paz/cameras')
distortion = np.loadtxt(filename)
camera = Camera(intrinsics, distortion)


model = KeypointNet2D(input_shape, num_keypoints)
model_name = '_'.join([model.name, str(num_keypoints), class_name])
filename = os.path.join(model_name, '_'.join([model_name, 'weights.hdf5']))
filename = get_file(filename, None, cache_subdir='paz/models')
model.load_weights(filename)

dimensions = {None: [.1, 0.08]}
pipeline = KeypointToPoseInference(model, points3D, camera, dimensions)
# video_player = VideoPlayer((1280, 960), pipeline, 0)
video_player = VideoPlayer((640, 480), pipeline, 0)
video_player.start()
