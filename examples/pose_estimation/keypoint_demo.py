import os
# import numpy as np
from paz.pipelines import KeypointInference
from paz.models import KeypointNet2D
# from paz.models import HRNetResidual
from paz.core import VideoPlayer
from tensorflow.keras.utils import get_file

num_keypoints, class_name, input_shape = 10, '035_power_drill', (128, 128, 3)
# model_name = '_'.join(['keypointnet-shared', str(num_keypoints), class_name])
# filename = os.path.join(model_name, 'keypoints_mean.txt')
# filename = get_file(filename, None, cache_subdir='paz/models')
# points3D = np.loadtxt(filename)[:, :3].astype(np.float64)

# filename = os.path.join('logitech_c270', 'camera_intrinsics.txt')
# filename = get_file(filename, None, cache_subdir='paz/cameras')
# camera_intrinsics = np.loadtxt(filename)
# filename = os.path.join('logitech_c270', 'distortions.txt')
# filename = get_file(filename, None, cache_subdir='paz/cameras')
# distortions = np.loadtxt(filename)


model = KeypointNet2D(input_shape, num_keypoints)
# model = HRNetResidual(input_shape, num_keypoints)
model_name = '_'.join([model.name, str(num_keypoints), class_name])
filename = os.path.join(model_name, '_'.join([model_name, 'weights.hdf5']))
filename = get_file(filename, None, cache_subdir='paz/models')
model.load_weights(filename)

pipeline = KeypointInference(model, num_keypoints, 5)

video_player = VideoPlayer((1280, 960), pipeline, 2)
video_player.run()
