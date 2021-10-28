import os
import cv2
import numpy as np
from paz.models import UNET_VGG16
from paz.backend.image import show_image, load_image
from paz import processors as pr
from paz.backend.camera import Camera
from scenes import PixelMaskRenderer
from processors import DrawBoxes3D
# from backend import homogenous_quaternion_to_rotation_matrix
from backend import solve_PnP_RANSAC
from backend import project_to_image
from backend import build_cube_points3D
from backend import draw_cube
from pipelines import Pix2Pose
from pipelines import EstimatePoseMasks
from paz.backend.camera import VideoPlayer
from paz.applications import SSD300FAT


image_shape = (128, 128, 3)
num_classes = 3

model = UNET_VGG16(num_classes, image_shape, freeze_backbone=True)
model.load_weights('UNET_weights_epochs-10_beta-3.hdf5')

# approximating intrinsic camera parameters
camera = Camera(device_id=0)
# camera.start()
# image_size = camera.read().shape[0:2]
# camera.stop()

image = load_image('test_image.jpg')
image_size = image.shape[0:2]
focal_length = image_size[1]
image_center = (image_size[1] / 2.0, image_size[0] / 2.0)
camera.distortion = np.zeros((4))
camera.intrinsics = np.array([[focal_length, 0, image_center[0]],
                              [0, focal_length, image_center[1]],
                              [0, 0, 1]])


object_sizes = np.array([0.184, 0.187, 0.052])
# epsilon = 0.005
epsilon = 0.15
detect = SSD300FAT(draw=False)
offsets = [0.1, 0.1]
estimate_keypoints = Pix2Pose(model, object_sizes)
pipeline = EstimatePoseMasks(detect, estimate_keypoints, camera, offsets, None)

results = pipeline(image)
predicted_image = results['image']
show_image(predicted_image)

# image_size = (640, 480)
# player = VideoPlayer(image_size, pipeline, camera)
# player.run()
"""
def show_results():
    image, alpha, RGB_mask_true = renderer.render()
    normalized_image = np.expand_dims(image / 255.0, 0)
    RGB_mask_pred = model.predict(normalized_image)
    RGB_mask_pred = np.squeeze(RGB_mask_pred, 0)
    RGB_mask_pred[RGB_mask_pred < epsilon] = 0.0
    show_image((RGB_mask_pred * 255.0).astype('uint8'))

    mask_pred = np.sum(RGB_mask_pred, axis=2)
    non_zero_arguments = np.nonzero(mask_pred)
    RGB_mask_pred = RGB_mask_pred[non_zero_arguments]
    RGB_mask_pred = (2.0 * RGB_mask_pred) - 1.0
    # this RGB mask scaling is good since you are scaling in RGB space
    object_points3D = (object_size / 2.0) * RGB_mask_pred
    num_points = len(object_points3D)

    row_args, col_args = non_zero_arguments
    row_args = row_args.reshape(-1, 1)
    col_args = col_args.reshape(-1, 1)
    image_points2D = np.concatenate([col_args, row_args], axis=1)
    image_points2D = image_points2D.reshape(num_points, 1, 2)
    image_points2D = image_points2D.astype(np.float64)
    image_points2D = np.ascontiguousarray(image_points2D)

    rotation_vector, translation = solve_PnP_RANSAC(
        object_points3D, image_points2D, camera.intrinsics)
    rotation_matrix = np.eye(3)
    cv2.Rodrigues(rotation_vector, rotation_matrix)
    translation = np.squeeze(translation, 1)
    points3D = build_cube_points3D(0.184, 0.187, 0.052)
    points2D = project_to_image(
        rotation_matrix, translation, points3D, camera.intrinsics)
    points2D = points2D.astype(np.int32)
    image = draw_cube(image.astype(float), points2D)
    image = image.astype('uint8')
    show_image(image)
"""
