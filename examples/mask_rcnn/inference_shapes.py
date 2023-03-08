from mask_rcnn.model import MaskRCNN
import tensorflow as tf
from mask_rcnn.utils import norm_boxes_graph
from mask_rcnn.inference_graph import InferenceGraph
from mask_rcnn.detection import ResizeImages, NormalizeImages
from mask_rcnn.detection import Detect, PostprocessInputs
from paz.abstract import SequentialProcessor
from paz.datasets.shapes import Shapes
import cv2
import utils
import numpy as np


def test(images, weights_path):
    resize = SequentialProcessor([ResizeImages(800, 0, 1024)])
    molded_images, windows = resize([images])
    image_shape = molded_images[0].shape
    window = norm_boxes_graph(windows[0], image_shape[:2])

    base_model = MaskRCNN(model_dir='../../mask_rcnn', image_shape=image_shape, backbone="resnet101",
                          batch_size=1, images_per_gpu=1, rpn_anchor_scales=(32, 64, 128, 256, 512),
                          train_rois_per_image=200, num_classes=81, window=window)

    inference_model = base_model.build_inference_model()

    base_model.keras_model = inference_model
    base_model.keras_model.load_weights(weights_path, by_name=True)
    preprocess = SequentialProcessor([ResizeImages(800, 0, 1024),
                                      NormalizeImages()])
    postprocess = SequentialProcessor([PostprocessInputs()])
    detect = Detect(base_model, (32, 64, 128, 256, 512), 1, preprocess, postprocess)
    results = detect([images])
    return results

path = ''  # Weights path to declare

config = TestConfig()
dataset_train = Shapes(1, (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]))
data = dataset_train.load_data()
images = data[0]['image']

class_names = ['BG', 'Square', 'Circle', 'Triangle']
results = test([images], path)
r = results[0]

utils.display_instances(images, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
