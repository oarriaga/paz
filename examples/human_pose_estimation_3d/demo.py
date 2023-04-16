from pipeline import SimpleBaselines
from paz.applications import HigherHRNetHumanPose2D
from paz.backend.image import load_image
from linear_model import Simple_Baseline
import numpy as np
from paz.backend.camera import Camera
from viz import visualize
from scipy.optimize import least_squares
from keypoints_processors import SolveTranslation3D


h36m_to_coco_joints2D = [4, 12, 14, 16, 11, 13, 15, 2, 1, 0, 5, 7, 9, 6, 8, 10]
args_to_joints3D = [0, 1, 2, 3, 6, 7, 8, 12, 13, 15, 17, 18, 19, 25, 26, 27]
args_to_mean = {1: [5, 6], 4: [11, 12], 2: [1, 4]}
path = 'test_image.jpg'
image = load_image(path)
image_height, image_width = image.shape[:2]
camera = Camera()
camera.intrinsics_from_HFOV(HFOV=70, image_shape=[image_height, image_width])
intrinsics = [camera.intrinsics[0, 0], np.array([[camera.intrinsics[0, 2],
                                                  camera.intrinsics[1, 2]]]
                                                ).flatten()]
keypoints2D = HigherHRNetHumanPose2D()
model = Simple_Baseline(16, 3, 1024, (32,), 2, 1)
model.load_weights('weights.h5')
pipeline = SimpleBaselines(model, args_to_mean, h36m_to_coco_joints2D)
keypoints = pipeline(image)
solvetranslation_processor = SolveTranslation3D(least_squares, intrinsics, args_to_joints3D)
joints3D, keypoints3D, poses3D = solvetranslation_processor(
                                 keypoints['keypoints2D'],
                                 keypoints['keypoints3D'])

visualize(keypoints['keypoints2D'], joints3D, keypoints3D, poses3D)
