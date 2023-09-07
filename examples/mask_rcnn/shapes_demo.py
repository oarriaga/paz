import cv2
import numpy as np
import tensorflow as tf

from paz.abstract import SequentialProcessor

from mask_rcnn.model.model import MaskRCNN
from mask_rcnn.datasets.shapes import Shapes

from mask_rcnn.pipelines.detection import ResizeImages, NormalizeImages
from mask_rcnn.pipelines.detection import Detect, PostprocessInputs

from mask_rcnn.inference import test
from mask_rcnn.utils import display_instances


image_min_dim = 128
image_max_dim = 128
image_scale = 0
anchor_ratios = (8, 16, 32, 64, 128)
images_per_gpu = 1
num_classes = 4

path = ''  # Weights path

dataset_train = Shapes(1, (128, 128))
data = dataset_train.load_data()
images = data[0]['input_image']

class_names = ['BG', 'Square', 'Circle', 'Triangle']
results = test(images, path, 32, num_classes, 1, images_per_gpu, anchor_ratios,
               [image_max_dim, image_min_dim], image_scale)
r = results[0]

print(r)
display_instances(images, r['rois'], r['masks'], r['class_ids'], class_names,
                  r['scores'])

