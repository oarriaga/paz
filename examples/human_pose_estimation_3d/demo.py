import os
import numpy as np
from paz.backend.camera import Camera
from paz.pipelines.keypoints import HRNetSimpleBaselines
from paz.backend.image import load_image
from paz.models import SimpleBaseline
from tensorflow.keras.utils import get_file
from scipy.optimize import least_squares
from paz.backend.keypoints import filter_keypoints3D
from paz.backend.keypoints import initialize_translation, solve_least_squares,\
    get_bones_length, compute_reprojection_error, compute_optimized_pose3D
from viz import visualize


args_to_joints3D = [0, 1, 2, 3, 6, 7, 8, 12, 13, 15, 17, 18, 19, 25, 26, 27]
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
focal_length = intrinsics[0]
image_center = intrinsics[1]
joints3D = filter_keypoints3D(keypoints['keypoints3D'], args_to_joints3D)
root2D = keypoints['keypoints2D'][:, :2]
length2D, length3D = get_bones_length(keypoints['keypoints2D'], joints3D)
ratio = length3D / length2D
initial_joint_translation = initialize_translation(focal_length, root2D,
                                                   image_center, ratio)
joint_translation = solve_least_squares(least_squares,
                                        compute_reprojection_error,
                                        initial_joint_translation,
                                        joints3D, keypoints['keypoints2D'],
                                        focal_length, image_center)
keypoints3D = np.reshape(keypoints['keypoints3D'], (-1, 32, 3))
optimized_poses3D = compute_optimized_pose3D(keypoints3D,
                                             joint_translation,
                                             focal_length, image_center)
visualize(keypoints['keypoints2D'], joints3D, keypoints3D, optimized_poses3D)
