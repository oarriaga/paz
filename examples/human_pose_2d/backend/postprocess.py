import numpy as np
import tensorflow as tf
from munkres import Munkres
import cv2


with_heatmap_loss = (True, True)
test_with_heatmap = (True, True)
with_AE_loss = (True, False)
test_with_AE = (True, False)
dataset = 'COCO'
dataset_with_centers = False
test_ignore_centers = True
test_scale_factor = [1]
test_project2image = True
test_flip_test = True
