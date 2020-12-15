import pytest
import numpy as np
import paz.backend as P
import paz.processors as pr
from paz.models.detection.utils import create_prior_boxes
from paz.abstract import SequentialProcessor

from mask_rcnn.config import Config
from mask_rcnn.pipeline import DetectionPipeline, DataSequencer


class YCBVideoConfig(Config):
    NAME = 'ycb'
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 21


@pytest.fixture
def prior_boxes():
    return create_prior_boxes('YCBVideo')


@pytest.fixture
def image_path():
    return '../example/test.png'


@pytest.fixture
def box_data(image_path):
    H, W = P.image.load_image(image_path).shape[:2]
    boxes = np.array([[160/W, 64/H, 222/W, 134/H, 1],
                      [29/W, 171/H, 107/W, 266/H, 2],
                      [83/W, 147/H, 201/W, 283/H, 3]])
    return boxes


@pytest.fixture
def mask_data():
    mask = {'12': '../example/32_12.png',
            '03': '../example/32_03.png',
            '02': '../example/32_02.png'}
    return mask


@pytest.fixture
def target_mask(mask_data):
    masks = P.image.load_mask(mask_data)
    return np.sum(masks)


@pytest.fixture
def class_names():
    classes = ['background', '021_bleach_cleanser', '004_sugar_box', 
               '003_cracker_box']
    return classes


@pytest.fixture
def augmentator(prior_boxes, class_names):
    config = YCBVideoConfig()
    return DetectionPipeline(config, prior_boxes,
                             num_classes=len(class_names))


@pytest.fixture
def data(image_path, box_data, mask_data):
    sample = [{'image': image_path, 'boxes': box_data, 'masks': mask_data}]
    return sample


@pytest.fixture
def boxes2D(prior_boxes, class_names):
    to_boxes2D = SequentialProcessor([
        pr.ControlMap(pr.DecodeBoxes(prior_boxes), [1], [1]),
        pr.ControlMap(pr.ToBoxes2D(class_names, True), [1], [1]),
        pr.ControlMap(pr.DenormalizeBoxes2D(), [0, 1], [1], {0: 0}),
        pr.ControlMap(pr.FilterClassBoxes2D(class_names[1:]), [1], [1])])
    return to_boxes2D


@pytest.fixture
def target_boxes():
    target_box = np.array([[186, 239, 319, 357],
                           [269, 283, 363, 364],
                           [291, 207, 454, 373]])
    return target_box


@pytest.fixture
def sequencer(augmentator, data):
    batch_size = 1
    sequencer = DataSequencer(augmentator, batch_size, data)
    batch = sequencer.__getitem__(0)
    batch_images = batch[0]['image']
    batch_boxes, batch_masks = batch[1]['boxes'], batch[1]['masks']
    image, boxes, masks = batch_images[0], batch_boxes[0], batch_masks[0]
    image = (image + pr.BGR_IMAGENET_MEAN).astype('uint8')
    image = P.image.convert_color_space(image, pr.BGR2RGB)
    return image, boxes, masks


@pytest.mark.parametrize('ones', [13735427.])
def test_masks(sequencer, ones):
    _, _, masks = sequencer
    assert np.sum(masks) == ones


def test_boxes(sequencer, boxes2D, class_names, target_boxes):
    image, boxes, _ = sequencer
    boxes2d = boxes2D(image, boxes)
    coordinates = []
    for box2d in boxes2d[1]:
        if box2d.class_name in class_names:
            coordinates.append(box2d.coordinates)
    unique_boxes = [value for value in {item for item in coordinates}]
    assert np.array(unique_boxes).all() == target_boxes.all()
