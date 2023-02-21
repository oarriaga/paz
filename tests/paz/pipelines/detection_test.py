import pytest
import numpy as np
import cv2

import os
from tensorflow.keras.utils import get_file
from paz.backend.image import load_image

from paz.pipelines import (
    SSD512COCO, SSD300VOC, SSD300FAT, SSD512YCBVideo, EFFICIENTDETD0COCO,
    EFFICIENTDETD1COCO, EFFICIENTDETD2COCO, EFFICIENTDETD3COCO,
    EFFICIENTDETD4COCO, EFFICIENTDETD5COCO, EFFICIENTDETD6COCO,
    EFFICIENTDETD7COCO)
from paz.pipelines import HaarCascadeFrontalFace
from paz.pipelines import DetectFaceKeypointNet2D32
from paz.pipelines import DetectMiniXceptionFER
from paz.abstract.messages import Box2D


@pytest.fixture
def image_with_everyday_objects():
    URL = ('https://github.com/oarriaga/altamira-data/releases/download'
           '/v0.9.1/image_with_everyday_classes.jpg')
    filename = os.path.basename(URL)
    fullpath = get_file(filename, URL, cache_subdir='paz/tests')
    image = load_image(fullpath)
    return image


@pytest.fixture
def image_with_tools():
    URL = ('https://github.com/oarriaga/altamira-data/releases/download'
           '/v0.9.1/image_with_tools.jpg')
    filename = os.path.basename(URL)
    fullpath = get_file(filename, URL, cache_subdir='paz/tests')
    image = load_image(fullpath)
    return image


@pytest.fixture
def image_with_faces():
    URL = ('https://github.com/oarriaga/altamira-data/releases/download'
           '/v0.9.1/image_with_faces.jpg')
    filename = os.path.basename(URL)
    fullpath = get_file(filename, URL, cache_subdir='paz/tests')
    image = load_image(fullpath)
    return image


@pytest.fixture
def image_with_multiple_objects():
    URL = ('https://github.com/oarriaga/altamira-data/releases/download/'
           'v0.16/image_with_multiple_objects.png')
    filename = os.path.basename(URL)
    fullpath = get_file(filename, URL, cache_subdir='paz/tests')
    image = load_image(fullpath)
    return image


@pytest.fixture
def boxes_SSD512COCO():
    boxes2D = [
        Box2D(np.array([544, 373, 1018, 807]), 0.9982471, 'person'),
        Box2D(np.array([483, 710, 569, 819]), 0.78569597, 'cup'),
        Box2D(np.array([943, 182, 1083, 341]), 0.7874794, 'potted plant'),
        Box2D(np.array([150, 721, 1413, 993]), 0.6786173, 'dining table'),
        Box2D(np.array([577, 619, 895, 820]), 0.83648031, 'laptop')]
    return boxes2D


@pytest.fixture
def boxes_SSD300VOC():
    boxes2D = [
        Box2D(np.array([510, 383, 991, 806]), 0.99694544, 'person'),
        Box2D(np.array([954, 192, 1072, 350]), 0.7211749, 'pottedplant')]
    return boxes2D


@pytest.fixture
def boxes_SSD300FAT():
    boxes2D = [
        Box2D(np.array([110, 96, 150, 125]), 0.626648, '007_tuna_fish_can'),
        Box2D(np.array([41, 93, 146, 167]), 0.7510558, '035_power_drill'),
        Box2D(np.array([171, 22, 227, 131]), 0.793466, '006_mustard_bottle'),
        Box2D(np.array([99, 6, 151, 107]), 0.50704032, '003_cracker_box')]
    return boxes2D


@pytest.fixture
def boxes_SSD512YCBVideo():
    boxes2D = [
        Box2D(np.array([115, 121, 215, 155]), 0.9056605, '011_banana'),
        Box2D(np.array([38, 93, 162, 164]), 0.839406490, '035_power_drill'),
        Box2D(np.array([173, 25, 217, 123]), 0.99889081, '006_mustard_bottle'),
        Box2D(np.array([93, 11, 148, 104]), 0.825154304, '003_cracker_box')]
    return boxes2D


@pytest.fixture
def boxes_HaarCascadeFace():
    boxes2D = [
        Box2D(np.array([855, 466, 974, 585]), 1.0, 'Face'),
        Box2D(np.array([701, 362, 827, 488]), 1.0, 'Face'),
        Box2D(np.array([488, 408, 612, 532]), 1.0, 'Face'),
    ]
    return boxes2D


@pytest.fixture
def boxes_MiniXceptionFER():
    boxes2D = [
        Box2D(np.array([855, 466, 974, 585]), 0.98590159, 'happy'),
        Box2D(np.array([701, 362, 827, 488]), 0.49472683, 'neutral'),
        Box2D(np.array([488, 408, 612, 532]), 0.28161105, 'sad'),
    ]
    return boxes2D


@pytest.fixture
def boxes_FaceKeypointNet2D32():
    boxes2D = [
        Box2D(np.array([855, 466, 974, 585]), 1.0, 'Face'),
        Box2D(np.array([701, 362, 827, 488]), 1.0, 'Face'),
        Box2D(np.array([488, 408, 612, 532]), 1.0, 'Face'),
    ]
    return boxes2D


def boxes_EFFICIENTDETD0COCO():
    boxes2D = [
        Box2D(np.array([208, 88, 625, 473]), 0.91767185, 'person'),
        Box2D(np.array([134, 65, 389, 263]), 0.93340992, 'dog'),
        Box2D(np.array([-1, 82, 157, 236]), 0.78445029, 'potted plant'),
        Box2D(np.array([27, 152, 197, 468]), 0.74786907, 'tv'),
        Box2D(np.array([178, 269, 304, 325]), 0.81141179, 'mouse'),
        Box2D(np.array([216, 301, 414, 473]), 0.80782127, 'keyboard')
        ]
    return boxes2D


def boxes_EFFICIENTDETD1COCO():
    boxes2D = [
        Box2D(np.array([205, 85, 632, 476]), 0.96687036, 'person'),
        Box2D(np.array([132, 58, 391, 265]), 0.97678065, 'dog'),
        Box2D(np.array([2, 81, 151, 244]), 0.74390614, 'potted plant'),
        Box2D(np.array([33, 159, 199, 455]), 0.88892513, 'tv'),
        Box2D(np.array([184, 271, 239, 305]), 0.81121546, 'mouse'),
        Box2D(np.array([176, 270, 302, 331]), 0.74208050, 'mouse'),
        Box2D(np.array([221, 306, 412, 477]), 0.95130372, 'keyboard')
        ]
    return boxes2D


def boxes_EFFICIENTDETD2COCO():
    boxes2D = [
        Box2D(np.array([226, 78, 627, 475]), 0.96944528, 'person'),
        Box2D(np.array([137, 61, 387, 263]), 0.89439338, 'dog'),
        Box2D(np.array([418, 113, 469, 215]), 0.65552788, 'chair'),
        Box2D(np.array([1, 86, 150, 230]), 0.71399605, 'potted plant'),
        Box2D(np.array([190, 273, 240, 306]), 0.83802855, 'mouse'),
        Box2D(np.array([255, 280, 303, 312]), 0.73921251, 'mouse'),
        Box2D(np.array([180, 270, 301, 325]), 0.62887740, 'mouse'),
        Box2D(np.array([221, 309, 411, 478]), 0.96415239, 'keyboard'),
        Box2D(np.array([13, 397, 169, 478]), 0.63690769, 'book')
        ]
    return boxes2D


def boxes_EFFICIENTDETD3COCO():
    boxes2D = [
        Box2D(np.array([200, 77, 628, 474]), 0.95468121, 'person'),
        Box2D(np.array([136, 61, 391, 261]), 0.97712504, 'dog'),
        Box2D(np.array([417, 112, 469, 216]), 0.77876096, 'chair'),
        Box2D(np.array([0, 84, 153, 220]), 0.88972324, 'potted plant'),
        Box2D(np.array([27, 151, 201, 466]), 0.84212630, 'tv'),
        Box2D(np.array([187, 274, 241, 306]), 0.91222530, 'mouse'),
        Box2D(np.array([258, 281, 304, 313]), 0.80773502, 'mouse'),
        Box2D(np.array([222, 307, 413, 477]), 0.94964569, 'keyboard'),
        Box2D(np.array([9, 396, 169, 476]), 0.69264298, 'book'),
        Box2D(np.array([460, 412, 483, 452]), 0.69317573, 'clock')
        ]
    return boxes2D


def boxes_EFFICIENTDETD4COCO():
    boxes2D = [
        Box2D(np.array([196, 80, 628, 476]), 0.99436223, 'person'),
        Box2D(np.array([136, 61, 389, 261]), 0.99237543, 'dog'),
        Box2D(np.array([417, 112, 468, 216]), 0.80065679, 'chair'),
        Box2D(np.array([0, 83, 152, 221]), 0.93566381, 'potted plant'),
        Box2D(np.array([29, 148, 198, 462]), 0.88566303, 'tv'),
        Box2D(np.array([185, 274, 243, 307]), 0.77240949, 'mouse'),
        Box2D(np.array([238, 279, 303, 311]), 0.72419583, 'mouse'),
        Box2D(np.array([219, 309, 409, 477]), 0.97060418, 'keyboard'),
        Box2D(np.array([10, 395, 170, 477]), 0.84335249, 'book'),
        Box2D(np.array([459, 411, 484, 451]), 0.70195990, 'clock')
        ]
    return boxes2D


def boxes_EFFICIENTDETD5COCO():
    boxes2D = [
        Box2D(np.array([183, 76, 632, 475]), 0.96840107, 'person'),
        Box2D(np.array([138, 61, 388, 259]), 0.98358279, 'dog'),
        Box2D(np.array([416, 112, 468, 217]), 0.74267524, 'chair'),
        Box2D(np.array([0, 83, 153, 221]), 0.87362897, 'potted plant'),
        Box2D(np.array([30, 159, 194, 459]), 0.89469951, 'tv'),
        Box2D(np.array([188, 274, 241, 306]), 0.88032329, 'mouse'),
        Box2D(np.array([247, 280, 304, 313]), 0.70261341, 'mouse'),
        Box2D(np.array([218, 310, 410, 477]), 0.98620319, 'keyboard')
        ]
    return boxes2D


def boxes_EFFICIENTDETD6COCO():
    boxes2D = [
        Box2D(np.array([133, 32, 688, 520]), 0.97538018, 'person'),
        Box2D(np.array([102, 34, 420, 281]), 0.98068982, 'dog'),
        Box2D(np.array([-20, 68, 171, 241]), 0.86188519, 'potted plant'),
        Box2D(np.array([11, 115, 222, 509]), 0.87279641, 'tv'),
        Box2D(np.array([182, 269, 249, 311]), 0.88012033, 'mouse'),
        Box2D(np.array([199, 288, 434, 496]), 0.98532408, 'keyboard')
        ]
    return boxes2D


def boxes_EFFICIENTDETD7COCO():
    boxes2D = [
        Box2D(np.array([196, 77, 630, 474]), 0.98574125, 'person'),
        Box2D(np.array([137, 63, 391, 260]), 0.99497270, 'dog'),
        Box2D(np.array([342, 111, 470, 365]), 0.83805811, 'chair'),
        Box2D(np.array([0, 84, 153, 221]), 0.96069687, 'potted plant'),
        Box2D(np.array([32, 155, 197, 466]), 0.93104147, 'tv'),
        Box2D(np.array([191, 274, 240, 306]), 0.94373917, 'mouse'),
        Box2D(np.array([244, 280, 304, 315]), 0.67447489, 'mouse'),
        Box2D(np.array([220, 310, 409, 477]), 0.99342632, 'keyboard')
        ]
    return boxes2D


def assert_inferences(detector, image, labelled_boxes):
    inferences = detector(image)
    predicted_boxes2D = inferences['boxes2D']
    assert len(predicted_boxes2D) == len(labelled_boxes)
    for box2D, predicted_box2D in zip(labelled_boxes, predicted_boxes2D):
        assert np.allclose(box2D.coordinates, predicted_box2D.coordinates)
        assert np.allclose(box2D.score, predicted_box2D.score)
        assert (box2D.class_name == predicted_box2D.class_name)


def test_SSD512COCO(image_with_everyday_objects, boxes_SSD512COCO):
    detector = SSD512COCO()
    assert_inferences(detector, image_with_everyday_objects, boxes_SSD512COCO)


def test_SSD300VOC(image_with_everyday_objects, boxes_SSD300VOC):
    detector = SSD300VOC()
    assert_inferences(detector, image_with_everyday_objects, boxes_SSD300VOC)


def test_SSD300FAT(image_with_tools, boxes_SSD300FAT):
    detector = SSD300FAT(0.5)
    assert_inferences(detector, image_with_tools, boxes_SSD300FAT)


def test_SSD512YCBVideo(image_with_tools, boxes_SSD512YCBVideo):
    detector = SSD512YCBVideo()
    assert_inferences(detector, image_with_tools, boxes_SSD512YCBVideo)


# TODO: OpenCV is not deterministic with it's output
def test_HaarCascadeFrontalFace(image_with_faces, boxes_HaarCascadeFace):
    cv2.ocl.setUseOpenCL(False)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(777)
    detector = HaarCascadeFrontalFace()
    assert_inferences(detector, image_with_faces, boxes_HaarCascadeFace)


def test_DetectMiniXceptionFER(image_with_faces, boxes_MiniXceptionFER):
    cv2.ocl.setUseOpenCL(False)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(777)
    detector = DetectMiniXceptionFER()
    assert_inferences(detector, image_with_faces, boxes_MiniXceptionFER)


def test_boxes_DetectFaceKeypointNet2D32(image_with_faces,
                                         boxes_FaceKeypointNet2D32):
    cv2.ocl.setUseOpenCL(False)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(777)
    detector = DetectFaceKeypointNet2D32()
    assert_inferences(detector, image_with_faces, boxes_FaceKeypointNet2D32)


@pytest.mark.parametrize(('detection_pipeline, boxes_EFFICIENTDETDXCOCO'),
                         [
                            (EFFICIENTDETD0COCO, boxes_EFFICIENTDETD0COCO),
                            (EFFICIENTDETD1COCO, boxes_EFFICIENTDETD1COCO),
                            (EFFICIENTDETD2COCO, boxes_EFFICIENTDETD2COCO),
                            (EFFICIENTDETD3COCO, boxes_EFFICIENTDETD3COCO),
                            (EFFICIENTDETD4COCO, boxes_EFFICIENTDETD4COCO),
                            (EFFICIENTDETD5COCO, boxes_EFFICIENTDETD5COCO),
                            (EFFICIENTDETD6COCO, boxes_EFFICIENTDETD6COCO),
                            (EFFICIENTDETD7COCO, boxes_EFFICIENTDETD7COCO),
                         ])
def test_EFFICIENTDETDXCOCO(
        detection_pipeline, image_with_multiple_objects,
        boxes_EFFICIENTDETDXCOCO):
    detector = detection_pipeline(score_thresh=0.60, nms_thresh=0.25)
    boxes_EFFICIENTDETDXCOCO = boxes_EFFICIENTDETDXCOCO()
    assert_inferences(
        detector, image_with_multiple_objects, boxes_EFFICIENTDETDXCOCO)
