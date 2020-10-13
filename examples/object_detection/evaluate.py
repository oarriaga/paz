from __future__ import division

import argparse

import numpy as np
from paz.datasets import VOC
from paz.datasets import get_class_names
from paz.evaluation import evaluateMAP
from paz.models import SSD300
from paz.pipelines import DetectSingleShot

class_names = get_class_names()
class_dict = {
    class_name: class_arg for class_arg, class_name in enumerate(class_names)
}

voc_root = './examples/object_detection/data/VOCdevkit'


def test(weights_path):
    """
    Arguments:
        weights_path: model path to be evaluated
    Returns:
        result: Dictionary of evaluation results
    """
    score_thresh, nms_thresh, labels = 0.01, .45, get_class_names('VOC')

    model = SSD300()
    model.load_weights(weights_path)
    detector = DetectSingleShot(model, labels, score_thresh, nms_thresh)

    data_name = 'VOC2007'
    data_split = 'test'
    data_manager = VOC(voc_root, data_split, name=data_name, evaluate=True)
    dataset = data_manager.load_data()

    result = evaluateMAP(
        detector,
        dataset,
        class_dict,
        iou_thresh=0.5,
        use_07_metric=True)

    result_str = "mAP: {:.4f}\n".format(result["map"])
    metrics = {'mAP': result["map"]}
    for arg, ap in enumerate(result["ap"]):
        if arg == 0 or np.isnan(ap):  # skip background
            continue
        metrics[class_names[arg]] = ap
        result_str += "{:<16}: {:.4f}\n".format(class_names[arg], ap)
    print(result_str)


description = 'Test script for single-shot object detection models'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-wp', '--weights_path', default=None,
                    type=str, help='Path for model to be evaluated')
args = parser.parse_args()

test(args.weights_path)
