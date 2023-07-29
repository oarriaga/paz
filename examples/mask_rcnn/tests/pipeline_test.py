import pytest

from paz.abstract import ProcessingSequence
from mask_rcnn.shapes_loader import Shapes

from mask_rcnn.pipelines.data_generator import ComputeBackboneShapes
from mask_rcnn.pipelines.data_generator import GeneratePyramidAnchors
from mask_rcnn.pipelines.data_generator import MaskRCNN_pipeline
from mask_rcnn.pipelines.data_generator import MakeRPNLabel
from mask_rcnn.pipelines.pipeline import DetectionPipeline
from mask_rcnn.pipelines.data_generator import GeneratePyramidAnchors

from paz.abstract import ProcessingSequence


@pytest.fixture
def data():
    size = (320, 320)
    shapes = Shapes(1, size)
    dataset = shapes.load_data()
    image = dataset[0]['input_image']
    mask = dataset[0]['input_gt_mask']
    box_data = dataset[0]['input_gt_bbox']
    class_ids = dataset[0]['input_gt_class_ids']
    data = [{'image': image, 'boxes': box_data, 'mask': mask,
             'class_ids': class_ids}]
    return data


@pytest.fixture
def anchors():
    anchor_scales = (8, 16, 32, 64, 128)
    backbone_shapes = ComputeBackboneShapes()("resnet101", [320, 320])
    anchors = GeneratePyramidAnchors()(anchor_scales, [0.5, 1, 2],
                                       backbone_shapes, [4, 8, 16, 32, 64], 1)
    return anchors


@pytest.fixture
def inference_pipeline(anchors, num_classes=4):
    inference_augmentator = DetectionPipeline(config, anchors, num_classes)
    return inference_augmentator


@pytest.fixture
def train_pipeline(anchors, num_classes=4):
    anchor_scales = (8, 16, 32, 64, 128)
    train_augmentator = MaskRCNN_pipeline([320, 320], anchor_scales,
                                          "resnet_101")
    return train_augmentator


@pytest.fixture
def inference_sequencer(data, inference_pipeline):
    batch_size = 1
    sequencer = ProcessingSequence(inference_pipeline, batch_size, data)
    return sequencer


@pytest.fixture
def train_sequencer(data, train_pipeline):
    batch_size = 1
    sequencer = ProcessingSequence(train_pipeline, batch_size, data)
    return sequencer


def test_inference_pipeline(inference_sequencer, anchors):
    size = (320, 320, 3)
    batch = inference_sequencer.__getitem__(0)
    batch_images = batch[0]['input_image']
    batch_boxes = batch[1]['input_gt_boxes']
    batch_masks = batch[1]['input_gt_mask']
    batch_box_deltas = batch[1]['box_deltas']
    batch_matches = batch[1]['matches']

    image, boxes, masks = batch_images[0], batch_boxes[0], batch_masks[0]
    assert image.shape == size == masks.shape
    assert boxes.shape == (3, 5)
    assert batch_box_deltas.shape[1:] == (256, 4)
    assert batch_matches.shape[1:][0] == anchors.shape[0]


def test_train_pipeline(train_sequencer, anchors):
    size = (320, 320, 3)
    batch = train_sequencer.__getitem__(0)
    batch_images = batch[0]['input_image']
    batch_boxes = batch[1]['input_gt_boxes']
    batch_masks = batch[1]['input_gt_mask']
    batch_box_deltas = batch[1]['box_deltas']
    batch_matches = batch[1]['matches']

    image, boxes, masks = batch_images[0], batch_boxes[0], batch_masks[0]
    assert image.shape == size == masks.shape
    assert boxes.shape == (3, 5)
    assert batch_box_deltas.shape[1:] == (256, 4)
    assert batch_matches.shape[1:][0] == anchors.shape[0]


def test_make_RPN_labels(data):
    image_shape = [320, 320]
    anchor_scales = (8, 16, 32, 64, 128)
    class_ids, boxes, RPN_match, RPN_val = MakeRPNLabel(
        image_shape, anchor_scales, "resnet_101")(data["class_ids"],
                                                  data["boxes"])
    assert boxes.shape == (3, 5)
    assert RPN_match.shape[1:] == (4092, 4)
    assert RPN_val.shape[1:] == (4348, 4)


@pytest.mark.parametrize('boxes', [[5, 10, 20, 30]])
def test_apply_box_refinement(boxes):
    ground_box = [10, 20, 30, 40]
    refinement = normalize_log_refinement(boxes, ground_box)
    assert refinement == [0.5, 0.5, 0.28768207245178085, 0.0]


def test_compute_anchor_boxes_overlaps(anchors, anchor_boxes_overlaps):
    overlaps, no_crowd_bool = anchor_boxes_overlaps
    assert anchors.shape[0] == overlaps.shape[0] == no_crowd_bool.shape[0]


def test_rpn_match(rpn_match):
    rpn, anchor_iou_max = rpn_match
    assert np.all(np.unique(rpn) == (-1, 0, 1))
    assert np.all(np.unique(anchor_iou_max) == (0, 1, 2))


def test_compute_rpn_box(rpn_box):
    box = rpn_box
    assert box.shape[0] == 256
