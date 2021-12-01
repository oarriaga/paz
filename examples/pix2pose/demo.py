import numpy as np
from paz.models import UNET_VGG16
from paz.backend.image import show_image, load_image
from paz.backend.camera import Camera
from paz.backend.camera import VideoPlayer
from paz.applications import SSD300FAT

# from pipelines import Pix2Pose
# from pipelines import EstimatePoseMasks
from pipelines import Pix2Pose
from pipelines import EstimatePoseMasks


image_shape = (128, 128, 3)
num_classes = 3

model = UNET_VGG16(num_classes, image_shape, freeze_backbone=True)
model.load_weights('weights/UNET_weights_epochs-10_beta-3.hdf5')

# approximating intrinsic camera parameters
camera = Camera(device_id=0)
# camera.start()
# image_size = camera.read().shape[0:2]
# camera.stop()

# image = load_image('test_image2.jpg')
image = load_image('images/test_image.jpg')
image_size = image.shape[0:2]
focal_length = image_size[1]
image_center = (image_size[1] / 2.0, image_size[0] / 2.0)
camera.distortion = np.zeros((4))
camera.intrinsics = np.array([[focal_length, 0, image_center[0]],
                              [0, focal_length, image_center[1]],
                              [0, 0, 1]])
# object_sizes = np.array([0.184, 0.187, 0.052])
epsilon = 0.001
score_thresh = 0.50
detect = SSD300FAT(score_thresh, draw=False)
offsets = [0.2, 0.2]
# estimate_keypoints = Pix2Pose(model, object_sizes, epsilon, True)
# pipeline = EstimatePoseMasks(detect, estimate_keypoints, camera, offsets)

object_sizes = np.array([1840, 1870, 520])
estimate_pose = Pix2Pose(model, object_sizes, camera, epsilon, draw=True)
# image = image[50:320, 60:320]
# show_image(estimate_pose(image)['image'])
pipeline = EstimatePoseMasks(detect, estimate_pose, offsets, True)
results = pipeline(image)
predicted_image = results['image']
show_image(predicted_image)

# image_size = (640, 480)
# player = VideoPlayer(image_size, pipeline, camera)
# player.run()
