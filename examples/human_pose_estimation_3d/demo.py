import os
import numpy as np
from scipy.optimize import least_squares
from paz.backend.camera import Camera
from paz.pipelines.keypoints import HRNetSimpleBaselines
from paz.backend.image import load_image
from paz.models import SimpleBaseline
from tensorflow.keras.utils import get_file
from paz.pipelines.keypoints import SolveTranslation3D
from paz.datasets.human36m import args_to_joints3D
from viz import visualize



URL = ('https://github.com/oarriaga/altamira-data/releases/download'
           '/v0.17/multiple_persons_posing.png')
filename = os.path.basename(URL)
fullpath = get_file(filename, URL, cache_subdir='paz/tests')
image = load_image(fullpath)
image_height, image_width = image.shape[:2]
camera = Camera()
camera.intrinsics_from_HFOV(HFOV=70, image_shape=[image_height, image_width])
intrinsics = [camera.intrinsics[0, 0], np.array([[camera.intrinsics[0, 2],
                                                  camera.intrinsics[1, 2]]]
                                                ).flatten()]
model = SimpleBaseline((32,), 16, 3, 1024, 2, 1, 'human36m')
pipeline = HRNetSimpleBaselines(model)
keypoints = pipeline(image)
keypoints3D = np.reshape(keypoints['keypoints3D'], (-1, 32, 3))
solveTranslation_pipeline = SolveTranslation3D(args_to_joints3D, intrinsics, least_squares)
joints3D, optimized_poses3D = solveTranslation_pipeline(keypoints)
visualize(keypoints['keypoints2D'], joints3D, keypoints3D, optimized_poses3D)
