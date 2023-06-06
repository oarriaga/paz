import os
from paz.backend.camera import Camera
from paz.backend.image import load_image
from scipy.optimize import least_squares
from tensorflow.keras.utils import get_file
from paz.pipelines import EstimateHumanPose
from paz.processors import OptimizeHumanPose3D
from paz.datasets.human36m import args_to_joints3D
from viz import visualize


URL = ('https://github.com/oarriaga/altamira-data/releases/download'
       '/v0.17/multiple_persons_posing.png')

filename = os.path.basename(URL)
fullpath = get_file(filename, URL, cache_subdir='paz/tests')
image = load_image(fullpath)
H, W = image.shape[:2]
camera = Camera()
camera.intrinsics_from_HFOV(HFOV=70, image_shape=[H, W])
pipeline = EstimateHumanPose()
optimize = OptimizeHumanPose3D(args_to_joints3D,
                               least_squares, camera.intrinsics)
keypoints = pipeline(image)
keypoints2D = keypoints['keypoints2D']
keypoints3D = keypoints['keypoints3D']
joints3D, optimized_poses3D = optimize(keypoints3D, keypoints2D)

visualize(keypoints2D, joints3D, keypoints3D, optimized_poses3D)
