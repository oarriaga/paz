import numpy as np
import tensorflow as tf
import os
import cv2

import mask_rcnn.model as modellib
from  mask_rcnn.config import Config
from mask_rcnn import utils
from mask_rcnn.inference_graph import InferenceGraph

MODEL_DIR = os.path.abspath("../../mask_rcnn")

class TestConfig(Config):
    NAME = "test"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80

def test(images):
    config = TestConfig()
    
    molded_image, window, _, _, _ = utils.resize_image(images[0],
                                min_dim=config.IMAGE_MIN_DIM,
                                min_scale=config.IMAGE_MIN_SCALE,
                                max_dim=config.IMAGE_MAX_DIM,
                                mode=config.IMAGE_RESIZE_MODE)
    
    image_shape = molded_image.shape
    window = utils.norm_boxes_graph(window, image_shape[:2])
    config.WINDOW = window
    
    base_model = modellib.MaskRCNN(config=config, model_dir=MODEL_DIR)

    #inference model
    inference_model = InferenceGraph(model=base_model, config=config)

    base_model.keras_model = inference_model()

    base_model.load_weights('../weights/mask_rcnn_coco.h5', by_name=True)

    results = base_model.detect(images, verbose=0)
    return results

#image = cv2.imread('../test_images/elephant.jpg')
#results = test(image)

