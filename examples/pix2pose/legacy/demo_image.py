import numpy as np
from paz.models import UNET_VGG16
from paz.backend.image import show_image, load_image
from paz.backend.camera import Camera
from paz.pipelines import DetectSingleShot
from paz.models import SSD300

from pipelines import MultiPix2Pose


image_path = 'images/lab_condition.png'
epsilon = 0.001
score_thresh = 0.50
offsets = [0.2, 0.2]
nms_thresh = 0.45

image_shape = (128, 128, 3)
num_classes = 3
camera = Camera(device_id=0)
image = load_image(image_path)
image_size = image.shape[0:2]
focal_length = image_size[1]
image_center = (image_size[1] / 2.0, image_size[0] / 2.0)
camera.distortion = np.zeros((4))
camera.intrinsics = np.array([[focal_length, 0, image_center[0]],
                              [0, focal_length, image_center[1]],
                              [0, 0, 1]])

class_names = ['background', 'Large_clamp', 'flat_screwdriver',
               'hammer', 'Solar_panel', 'power_drill']
detection = SSD300(len(class_names), head_weights=None)
detection.load_weights('weights/SSD300_weights_.53-1.40.hdf5')
detect = DetectSingleShot(detection, class_names, score_thresh,
                          nms_thresh, draw=False)

name_to_sizes = {
    'power_drill': np.array([1840, 1870, 520]),
    'Solar_panel': np.array([15000, 15000, 2000]),
    'Large_clamp': np.array([12000, 17100, 3900]),
    'hammer': np.array([18210, 33272, 3280])}


name_to_weights = {
    'power_drill': 'weights/UNET_weights_epochs-10_beta-3.hdf5',
    'Solar_panel': 'weights/UNET-VGG_solar_panel_canonical_13.hdf5',
    'Large_clamp': 'weights/UNET-VGG_large_clamp_canonical_10.hdf5',
    'hammer': 'weights/UNET-VGG16_weights_hammer_10.hdf5'}


segment = UNET_VGG16(num_classes, image_shape, freeze_backbone=True)
valid_class_names = ['power_drill', 'Solar_panel', 'Large_clamp', 'hammer']

pipeline = MultiPix2Pose(detect, segment, camera, name_to_weights,
                         name_to_sizes, valid_class_names, offsets,
                         epsilon, draw=True)

results = pipeline(image)
predicted_image = results['image']
show_image(predicted_image)
