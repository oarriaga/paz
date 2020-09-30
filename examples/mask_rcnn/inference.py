import numpy as np
import tensorflow as tf

from mask_rcnn.model import MaskRCNN
from mask_rcnn.config import Config
from mask_rcnn.utils import norm_boxes_graph, resize_image
from mask_rcnn.inference_graph import InferenceGraph


class TestConfig(Config):
    NAME = "test"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80


def test(images, weights_path):
    config = TestConfig()
    molded_image, window, _, _, _ = resize_image(images[0],
                                min_dim=config.IMAGE_MIN_DIM,
                                min_scale=config.IMAGE_MIN_SCALE,
                                max_dim=config.IMAGE_MAX_DIM,
                                mode=config.IMAGE_RESIZE_MODE)
    image_shape = molded_image.shape
    window = norm_boxes_graph(window, image_shape[:2])
    config.WINDOW = window

    base_model = MaskRCNN(config=config, model_dir='../../mask_rcnn')
    inference_model = InferenceGraph(model=base_model, config=config)
    base_model.keras_model = inference_model()
    base_model.load_weights(weights_path, by_name=True)

    results = base_model.detect(images)
    return results

