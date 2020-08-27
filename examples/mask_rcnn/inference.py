import numpy as np
import os
import cv2

import mask_rcnn.model as modellib
from  mask_rcnn.config import Config
import mask_rcnn.utils as mrcnn_utils


MODEL_DIR = os.path.abspath("../../mask_rcnn")

class TestConfig(Config):
    NAME = "test"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80

def test(image):
    #define model
    model = modellib.MaskRCNN(mode='inference', config=TestConfig(), model_dir=MODEL_DIR)

    # load coco model weights
    model.load_weights('/home/incendio/Documents/Thesis/mask_rcnn_coco.h5', by_name=True)

    #prediction
    results = model.detect([image], verbose=0)
    return results

#image = cv2.imread('../test_images/elephant.jpg')
#results = test(image)



