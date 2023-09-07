import pytest

import numpy as np
from paz.abstract import ProcessingSequence
from mask_rcnn.datasets.shapes import Shapes

from mask_rcnn.pipelines.data_generator import ComputeBackboneShapes
from mask_rcnn.pipelines.data_generator import GeneratePyramidAnchors
from mask_rcnn.pipelines.data_generator import MaskRCNNPipeline
from mask_rcnn.pipelines.data_generator import MakeRPNLabel
from mask_rcnn.pipelines.data_generator import GeneratePyramidAnchors

from paz.abstract import ProcessingSequence


@pytest.fixture
def data():
    size = (128, 128)
    shapes = Shapes(1, size)
    dataset = shapes.load_data()
    return dataset


@pytest.fixture
def anchors():
    anchor_scales = (8, 16, 32, 64, 128)
    backbone_shapes = ComputeBackboneShapes()("resnet101", (128, 128))
    anchors = GeneratePyramidAnchors()(anchor_scales, backbone_shapes,
                                       [0.5, 1, 2], [4, 8, 16, 32, 64])
    return anchors


@pytest.fixture
def train_pipeline():
    anchor_scales = (8, 16, 32, 64, 128)
    train_augmentator = MaskRCNNPipeline([128, 128, 3], anchor_scales,
                                         "resnet101")
    return train_augmentator


@pytest.fixture
def train_sequencer(data, train_pipeline):
    batch_size = 1
    sequencer = ProcessingSequence(train_pipeline, batch_size, data)
    return sequencer


def test_train_pipeline(train_sequencer, anchors):
    image_shape = (128, 128, 3)
    mask_shape = (128, 128, 100)
    batch = train_sequencer.__getitem__(0)
    batch_images = batch[0]['input_image']
    batch_boxes = batch[0]['input_gt_boxes']
    batch_masks = batch[0]['input_gt_masks']
    batch_box_deltas = batch[1]['rpn_bounding_box']
    batch_matches = batch[1]['rpn_class_logits']

    assert batch_images[0].shape == image_shape
    assert batch_masks[0].shape == mask_shape
    assert batch_boxes[0].shape == (100, 4)
    assert batch_box_deltas.shape[1:] == (4348, 4)
    assert batch_matches.shape[1:][0] == anchors.shape[0]


def test_make_RPN_labels(data):
    image_shape = [320, 320]
    anchor_scales = (8, 16, 32, 64, 128)
    class_ids, boxes, RPN_match, RPN_val = MakeRPNLabel(
        image_shape, anchor_scales, "resnet101")(data[0]["input_gt_class_ids"],
                                                 data[0]["input_gt_boxes"])

    assert class_ids.shape == data[0]["input_gt_class_ids"].shape
    assert boxes.shape == data[0]["input_gt_boxes"].shape
    assert RPN_match.shape == (25575, 1)
    assert RPN_val.shape == (25831, 4)
