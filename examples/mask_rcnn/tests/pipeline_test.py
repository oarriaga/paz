import pytest
from mask_rcnn.pipeline import DetectionPipeline
from mask_rcnn.shapes_loader import Shapes
from mask_rcnn.utils import compute_backbone_shapes
from mask_rcnn.utils import generate_pyramid_anchors
from paz.abstract import ProcessingSequence


@pytest.fixture
def data():
    size = (320, 320)
    shapes = Shapes(1, size)
    dataset = shapes.load_data()
    image = dataset[0]['image']
    mask = dataset[0]['mask']
    box_data = dataset[0]['box_data']
    data = [{'image': image, 'boxes': box_data, 'mask': mask}]
    return data


@pytest.fixture
def anchors():
    anchor_scales = (8, 16, 32, 64, 128)
    backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
    anchors = generate_pyramid_anchors(anchor_scales, [0.5, 1, 2],
                                       backbone_shapes, [4, 8, 16, 32, 64], 1)
    return anchors


@pytest.fixture
def pipeline(anchors, num_classes=4):
    config = ShapesConfig()
    augmentator = DetectionPipeline(config, anchors,
                                    num_classes=num_classes)
    return augmentator


@pytest.fixture
def sequencer(data, pipeline):
    batch_size = 1
    sequencer = ProcessingSequence(pipeline, batch_size, data)
    return sequencer


def test_pipeline(sequencer, anchors):
    size = (320, 320, 3)
    batch = sequencer.__getitem__(0)
    batch_images, batch_boxes = batch[0]['image'], batch[1]['boxes']
    batch_masks = batch[1]['mask']
    batch_box_deltas, batch_matches = batch[1]['box_deltas'], batch[1]['matches']
    image, boxes, masks = batch_images[0], batch_boxes[0], batch_masks[0]
    assert image.shape == size == masks.shape
    assert boxes.shape == (3, 5)
    assert batch_box_deltas.shape[1:] == (256, 4)
    assert batch_matches.shape[1:][0] == anchors.shape[0]


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
