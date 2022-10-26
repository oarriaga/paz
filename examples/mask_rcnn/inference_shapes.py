from mask_rcnn.model import MaskRCNN
from config import Config
from mask_rcnn.utils import norm_boxes_graph
from mask_rcnn.inference_graph import InferenceGraph
from mask_rcnn.detection import ResizeImages, NormalizeImages
from mask_rcnn.detection import Detect, PostprocessInputs
from paz.abstract import SequentialProcessor
from paz.datasets.shapes import Shapes
import cv2
import utils
import numpy as np

from keras.backend import manual_variable_initialization
manual_variable_initialization(True)
import tensorflow.compat.v1 as tf

class TestConfig(Config):
    NAME = "test"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 3
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128


def test(images, weights_path):
    config = TestConfig()
    resize = SequentialProcessor([ResizeImages(config)])
    molded_images, windows = resize(images)
    image_shape = molded_images[0].shape
    window = norm_boxes_graph(windows[0], image_shape[:2])
    config.WINDOW = window
    train_bn= config.TRAIN_BN
    image_shape= config.IMAGE_SHAPE
    backbone= config.BACKBONE
    top_down_pyramid_size= config.TOP_DOWN_PYRAMID_SIZE

    base_model = MaskRCNN(config=config, model_dir='../../mask_rcnn', train_bn=train_bn, image_shape=image_shape,
                          backbone=backbone, top_down_pyramid_size=top_down_pyramid_size)
    inference_model = InferenceGraph(model=base_model, config=config)

    base_model.keras_model = inference_model()

    tf.keras.Model.load_weights(base_model.keras_model, weights_path, by_name=True)

    preprocess = SequentialProcessor([ResizeImages(config),
                                      NormalizeImages(config)])
    postprocess = SequentialProcessor([PostprocessInputs()])
    detect = Detect(base_model, config, preprocess, postprocess)

    results = detect(images)
    return results


path = '' #Weights path to declare

config = TestConfig()
dataset_train = Shapes(1, (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]))
data = dataset_train.load_data()
images = data[0]['image']

class_names = ['BG', 'Square', 'Circle', 'Triangle']
results = test([images], path)
r = results[0]

utils.display_instances(images, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])