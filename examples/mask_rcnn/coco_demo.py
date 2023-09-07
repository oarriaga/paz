import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import get_file
import cv2

from paz.abstract import SequentialProcessor
from paz.backend.image.opencv_image import load_image

from mask_rcnn.model.model import MaskRCNN
from mask_rcnn.datasets.shapes import Shapes

from mask_rcnn.pipelines.detection import ResizeImages, NormalizeImages
from mask_rcnn.pipelines.detection import Detect, PostprocessInputs

from mask_rcnn.inference import test
from mask_rcnn.utils import display_instances

image_min_dim = 800
image_max_dim = 1024
image_scale = 0
anchor_ratios = (32, 64, 128, 256, 512)
images_per_gpu = 1
num_classes = 81

url = 'https://github.com/oarriaga/altamira-data/releases/tag/v0.18/'

weights_local_path = os.path.join(os.getcwd() + '/mask_rcnn_coco.h5')
image_local_path = os.path.join(os.getcwd() + '/television.jpeg')

weights_path = get_file(weights_local_path, url + '/mask_rcnn_coco.h5')
image_path = get_file(image_local_path, url + '/television.jpeg')

image = load_image(image_path)

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

results = test(image, weights_path, 128, num_classes, 1, images_per_gpu,
               anchor_ratios, [1024, 1024], 1)
r = results[0]
print(r)
display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names,
                  r['scores'])
