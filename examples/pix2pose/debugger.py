import os
import numpy as np
from paz.backend.image import show_image, resize_image
from paz.backend.camera import Camera
# from paz.pipelines.pose import RGBMaskToPose6D
from paz.pipelines import SingleInstancePIX2POSE6D
from paz.models.segmentation import UNET_VGG16
from scenes import PixelMaskRenderer
from paz.backend.image import (
    blend_alpha_channel, make_random_plain_image, concatenate_alpha_mask)

root_path = os.path.expanduser('~')
num_occlusions = 1
image_shape = (128, 128, 3)
viewport_size = image_shape[:2]
y_fov = 3.14159 / 4.0
light = [1.0, 30]
top_only = False
roll = 3.14159
shift = 0.05

# path_OBJ = '/home/octavio/052_extra_large_clamp_rotated/textured.obj'
OBJ_name = '.keras/paz/datasets/ycb_models/037_scissors/textured.obj'
path_OBJ = os.path.join(root_path, OBJ_name)
# path_OBJ = '/home/octavio/.keras/paz/datasets/ycb_models/035_power_drill/textured.obj'
distance = [0.30, 0.35]

renderer = PixelMaskRenderer(path_OBJ, viewport_size, y_fov, distance,
                             light, top_only, roll, shift)

camera = Camera()
camera.intrinsics_from_HFOV(image_shape=(128, 128))
# from meters to milimiters
# object_sizes = renderer.mesh.mesh.extents * 100
object_sizes = renderer.mesh.mesh.extents * 10000
print(object_sizes)
model = UNET_VGG16(3, image_shape, freeze_backbone=True)
model.load_weights('experiments/UNET-VGG16_RUN_00_04-04-2022_12-29-44/model_weights.hdf5')
# model.load_weights('experiments/UNET-VGG16_RUN_00_06-04-2022_11-20-18/model_weights.hdf5')
# model.load_weights('experiments/UNET-VGG16_RUN_00_07-04-2022_13-28-04/model_weights.hdf5')
# estimate_pose = RGBMaskToPose6D(model, object_sizes, camera, draw=True)
# estimate_pose = SingleInferencePIX2POSE6D(model, object_sizes, camera)
estimate_pose = SingleInstancePIX2POSE6D(model, object_sizes, camera)

image, alpha, RGBA_mask = renderer.render()
image = np.copy(image)  # TODO: renderer outputs unwritable numpy arrays
show_image(image)
results = estimate_pose(image)
show_image(results['image'])


for arg in range(100):
    image, alpha, RGBA_mask = renderer.render()
    RGBA = concatenate_alpha_mask(image, alpha)
    background = make_random_plain_image(image.shape)
    image_with_background = blend_alpha_channel(RGBA, background)
    results = estimate_pose(image_with_background.copy())
    image = np.concatenate(
        [image_with_background, RGBA_mask[..., 0:3], results['image']], axis=1)
    H, W = image.shape[:2]
    image = resize_image(image, (W * 3, H * 3))
    show_image(image)
