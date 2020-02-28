from __future__ import division

import os
import xml.etree.ElementTree as ET

import numpy as np
from paz.core import ops
from paz.datasets import VOC
from paz.datasets import get_class_names
from paz.models import SSD300
from paz.pipelines import SingleShotInference

class_names = get_class_names()
class_dict = {
    class_name: class_arg for class_arg, class_name in enumerate(class_names)
}

voc_root = './examples/object_detection/data/VOCdevkit'
score_thresh, nms_thresh, labels = 0.01, .45, get_class_names('VOC')
model = SSD300()
detector = SingleShotInference(model, labels, score_thresh, nms_thresh)

data_name = 'VOC2007'
data_split = 'test'
data_manager = VOC(voc_root, data_split, name=data_name, evaluate=True)
dataset = data_manager.load_data()

result = ops.evaluate_VOC(
            detector,
            dataset,
            class_dict,
            iou_thresh=0.5,
            use_07_metric=True)

result_str = "mAP: {:.4f}\n".format(result["map"])
metrics = {'mAP': result["map"]}
for arg, ap in enumerate(result["ap"]):
    if arg == 0:  # skip background
        continue
    metrics[class_names[arg]] = ap
    result_str += "{:<16}: {:.4f}\n".format(class_names[arg], ap)
print(result_str)
