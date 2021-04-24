import pytest
from mask_rcnn.pipeline import DetectionPipeline
from mask_rcnn.shapes_loader import Shapes
from mask_rcnn.config import Config
from mask_rcnn.utils import compute_backbone_shapes
from mask_rcnn.utils import generate_pyramid_anchors
from paz.abstract import ProcessingSequence


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    """
    NAME = 'shapes'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 3
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 320
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32


@pytest.fixture
def data():
    config = ShapesConfig()
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
    config = ShapesConfig()
    backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
    anchors = generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                       config.RPN_ANCHOR_RATIOS,
                                       backbone_shapes,
                                       config.BACKBONE_STRIDES,
                                       config.RPN_ANCHOR_STRIDE)
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
    config = ShapesConfig()
    batch = sequencer.__getitem__(0)
    batch_images, batch_boxes = batch[0]['image'], batch[1]['boxes']
    batch_masks = batch[1]['mask']
    batch_box_deltas, batch_matches = batch[1]['box_deltas'], batch[1]['matches']
    image, boxes, masks = batch_images[0], batch_boxes[0], batch_masks[0]
    assert image.shape == size == masks.shape
    assert boxes.shape == (3, 5)
    assert batch_box_deltas.shape[1:] == (config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4)
    assert batch_matches.shape[1:][0] == anchors.shape[0]

