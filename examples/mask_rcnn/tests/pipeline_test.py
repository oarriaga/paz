import pytest
import cv2
import numpy as np
from paz.models.detection.utils import create_prior_boxes
from mask_rcnn.pipeline import DetectionPipeline


@pytest.fixture
def image():
    image = cv2.imread('../example/image.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


@pytest.fixture
def boxes():
    box_data = np.array([[0., 0.5343, 0.2062, 0.8062, 2.],
                         [0.3249, 0.4718, 0.625, 0.7312, 3.],
                         [0.5531, 0.4218, 0.8093, 0.6437, 3.]])
    return box_data


@pytest.fixture
def masks():
    mask_path = ['../example/mask_0.png',
                 '../example/mask_1.png',
                 '../example/mask_2.png']
    masks = []
    for path in mask_path:
        masks.append(cv2.imread(path)[:, :, 0])
    masks = np.stack(masks, axis=2)
    return masks


@pytest.fixture
def class_names():
    classes = ['background', 'square', 'circle',
               'triangle']
    return classes


@pytest.fixture
def pipeline(class_names):
    return DetectionPipeline(create_prior_boxes(),
                             num_classes=len(class_names))


def test_pipeline(pipeline, image, boxes, masks):
    sample = {'image': image, 'boxes': boxes, 'mask': masks}
    data = pipeline(sample)
    assert (image.shape == data['inputs']['image'].shape)
    assert masks.shape == data['labels']['mask'].shape
