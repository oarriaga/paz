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
voc_root = './examples/object_detection/VOCdevkit'


def get_annotation(image_id):
    """
    Arguments:
        image_id: Int. ID of the image for which ground truth is needed

    Returns:
        boxes: numpy array, bounding boxes of the image
        labels: numpy array, labels corresponding the bounding box
        is_difficults: numpy array, Contains information whether the
        bounding box is difficult or not

    """
    annotation_file = os.path.join(
        voc_root, 'VOC2007', 'Annotations', '%s.xml' % image_id)
    objects = ET.parse(annotation_file).findall('object')
    boxes, labels, is_difficult = [], [], []
    for obj in objects:
        class_name = obj.find('name').text.lower().strip()
        bbox = obj.find('bndbox')
        # VOC dataset format follows Matlab, in which indexes start from 0
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1
        boxes.append([x1, y1, x2, y2])
        labels.append(class_dict[class_name])
        is_difficult_str = obj.find('difficult').text
        is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

    return (np.array(boxes, dtype=np.float32),
            np.array(labels, dtype=np.int64),
            np.array(is_difficult, dtype='bool'))


def get_predictions(dataset, detector):
    """
    Arguments:
        dataset: List containing information of the images from the
        Test dataset
        detector : Object for inference

    Returns:
        predictions_boxes: List containing prediction boxes
        predictions_labels: List containing corresponding prediction labels
        predictions_scores: List containing corresponding prediction scores
    """
    boxes, labels, scores = [], [], []
    for image in dataset:
        frame = ops.load_image(image['image'])
        results = detector({'image': frame})
        box, label, score = [], [], []
        for box2D in results['boxes2D']:
            score.append(box2D.score)
            box.append(list(box2D.coordinates))
            label.append(class_dict[box2D.class_name])
        boxes.append(np.array(box, dtype=np.float32))
        labels.append(np.array(label))
        scores.append(np.array(score, dtype=np.float32))
    return boxes, labels, scores


def get_ground_truths(dataset):
    """
    Arguments:
        dataset: List containing information of the images from the
        Test dataset

    Returns:
        ground_truth_boxes: List containing ground truth boxes
        ground_truth_labels: List containing corresponding ground truth labels
        ground_truth_difficults: List containing corresponding
        ground truth difficults
    """
    boxes, labels, difficults = [], [], []
    for image in dataset:
        image_id = image['image'].split('/')[-1].split('.')[0]
        box, label, difficult = get_annotation(image_id)
        boxes.append(box)
        labels.append(label)
        difficults.append(difficult)
    return boxes, labels, difficults


score_thresh, nms_thresh, labels = 0.01, .45, get_class_names('VOC')
model = SSD300()
detector = SingleShotInference(model, labels, score_thresh, nms_thresh)

data_names = [['VOC2007', 'VOC2012'], 'VOC2007']
data_splits = [['trainval', 'trainval'], 'test']

data_managers, datasets = [], []

for data_name, data_split in zip(data_names, data_splits):
    data_manager = VOC(voc_root, data_split, name=data_name)
    data_managers.append(data_manager)
    datasets.append(data_manager.load_data())


predictions_boxes, predictions_labels, predictions_scores = get_predictions(datasets[1], detector)
ground_truth_boxes, ground_truth_labels, ground_truth_difficults = get_ground_truths(datasets[1])

result = ops.evaluate_VOC(
            detector,
            dataset,
            # predictions_boxes,
            # predictions_labels,
            # predictions_scores,
            # ground_truth_boxes,
            # ground_truth_labels,
            # ground_truth_difficults,
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
