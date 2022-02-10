import numpy as np
from paz.models import UNET_VGG16
from paz.backend.image import show_image, load_image
from paz.backend.camera import Camera
from paz.backend.camera import VideoPlayer
from paz.applications import SSD300FAT

from pipelines import Pix2Pose, EstimatePoseMasks


image_shape = (128, 128, 3)
num_classes = 3

model = UNET_VGG16(num_classes, image_shape, freeze_backbone=True)
model.load_weights('experiments/UNET-VGG16_RUN_00_08-02-2022_14-39-55/weights.hdf5')
# model.load_weights('weights/UNET_weights_epochs-10_beta-3.hdf5')
# model.load_weights('weights/UNET-VGG_solar_panel_canonical_13.hdf5')
# model.load_weights('weights/UNET-VGG_large_clamp_canonical_10.hdf5')

# approximating intrinsic camera parameters
camera = Camera(device_id=0)
# camera.start()
# image_size = camera.read().shape[0:2]
# camera.stop()

image = load_image('images/test_image2.jpg')
# image = load_image('images/lab_condition.png')
image_size = image.shape[0:2]
focal_length = image_size[1]
image_center = (image_size[1] / 2.0, image_size[0] / 2.0)
camera.distortion = np.zeros((4))
camera.intrinsics = np.array([[focal_length, 0, image_center[0]],
                              [0, focal_length, image_center[1]],
                              [0, 0, 1]])
# object_sizes = np.array([0.184, 0.187, 0.052])
# object_sizes = np.array([184, 187, 52])
object_sizes = np.array([1840, 1870, 520])  # power drill
epsilon = 0.15
score_thresh = 0.50
detect = SSD300FAT(score_thresh, draw=False)
offsets = [0.5, 0.5]
estimate_keypoints = Pix2Pose(model, object_sizes, camera, epsilon, draw=False)
pipeline = EstimatePoseMasks(detect, estimate_keypoints, offsets)
predicted_image = pipeline(image)['image']
show_image(predicted_image)
from paz.backend.image import write_image
write_image('images/predicted_power_drill.png', predicted_image)

# object_sizes = np.array([1840, 1870, 520])  # power drill
# object_sizes = np.array([15000, 15000, 2000])  # solar panel
# object_sizes = np.array([15000, 15000, 2000])  # solar panel
# estimate_pose = Pix2Pose(model, object_sizes, camera, epsilon, draw=True)
# image = image[768:1324, 622:784]
# image = image[622:784, 768:1324]


# image_hammer = image[460:1030, 740:1340]
# model.load_weights('weights/UNET-VGG16_weights_hammer_10.hdf5')
# show_image(estimate_pose(image_hammer)['image'])

# show_image(image)
# image_clamp = image[670:1000, 1000:1400]
# image_hammer = image[460:1030, 740:1340]
# model.load_weights('weights/UNET-VGG_large_clamp_canonical_10.hdf5')
# show_image(estimate_pose(image_clamp)['image'])

"""
image = load_image('images/zed_left_1011.png')
image = image[250:800, 250:850, :]
H, W, num_channels = image.shape
show_image(estimate_pose(image)['image'])

image = load_image('images/MicrosoftTeams-image.png')
show_image(estimate_pose(image)['image'])

image = load_image('images/zed_left_705.png')
image = image[250:1080, 250:1400, :]
show_image(estimate_pose(image)['image'])


image = load_image('images/zed_left_792.png')
image = image[30:1400, 280:1060, :]
show_image(estimate_pose(image)['image'])
"""

# image = load_image('images/large_clamp.jpeg')
# show_image(image[1])
# results = estimate_pose(image)
# show_image(results['image'])


# pipeline = EstimatePoseMasks(detect, estimate_pose, offsets, True)
# results = pipeline(image)
# predicted_image = results['image']
# show_image(predicted_image)

# image_size = (640, 480)
# player = VideoPlayer(image_size, pipeline, camera)
# player.run()
