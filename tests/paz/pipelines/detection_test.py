import pytest
import numpy as np
import cv2

import os
from tensorflow.keras.utils import get_file
from paz.backend.image import load_image

from paz.pipelines import SSD512COCO, SSD300VOC, SSD300FAT, SSD512YCBVideo
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
