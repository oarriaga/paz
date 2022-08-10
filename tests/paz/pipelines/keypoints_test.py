import pytest
import numpy as np
import cv2

import os
from tensorflow.keras.utils import get_file
from paz.backend.image import load_image

from paz.pipelines import DetectFaceKeypointNet2D32
from paz.pipelines import FaceKeypointNet2D32


@pytest.fixture
def image_with_faces():
    URL = ('https://github.com/oarriaga/altamira-data/releases/download'
           '/v0.9.1/image_with_faces.jpg')
    filename = os.path.basename(URL)
    fullpath = get_file(filename, URL, cache_subdir='paz/tests')
    image = load_image(fullpath)
    return image


@pytest.fixture
def image_with_face():
    URL = ('https://github.com/oarriaga/altamira-data/releases/download'
           '/v0.9.1/image_with_face.jpg')
    filename = os.path.basename(URL)
    fullpath = get_file(filename, URL, cache_subdir='paz/tests')
    image = load_image(fullpath)
    return image


@pytest.fixture
def keypoints_DetectFaceKeypointNet2D32():
    keypoints_A = np.array(
        [[787., 412.],
         [742., 413.],
         [778., 411.],
         [797., 409.],
         [750., 413.],
         [732., 414.],
         [773., 402.],
         [805., 400.],
         [752., 404.],
         [723., 407.],
         [765., 443.],
         [791., 457.],
         [747., 463.],
         [768., 458.],
         [766., 464.]])

    keypoints_B = np.array(
        [[573., 456.],
         [526., 459.],
         [563., 456.],
         [581., 455.],
         [534., 459.],
         [516., 460.],
         [557., 446.],
         [589., 444.],
         [537., 449.],
         [507., 452.],
         [554., 485.],
         [579., 502.],
         [531., 507.],
         [554., 503.],
         [552., 508.]])

    keypoints_C = np.array(
        [[938., 514.],
         [892., 514.],
         [928., 514.],
         [946., 513.],
         [901., 515.],
         [883., 515.],
         [922., 505.],
         [954., 503.],
         [903., 508.],
         [874., 509.],
         [913., 545.],
         [941., 560.],
         [895., 564.],
         [916., 562.],
         [914., 567.]])
    keypoints2D = [keypoints_C, keypoints_A, keypoints_B]
    return keypoints2D


@pytest.fixture
def keypoints_FaceKeypointNet2D32():
    keypoints = np.array(
        [[183., 129.],
         [77., 135.],
         [160., 129.],
         [198., 121.],
         [100., 135.],
         [58., 141.],
         [146., 106.],
         [218., 96.],
         [103., 114.],
         [39., 119.],
         [134., 209.],
         [199., 245.],
         [92., 264.],
         [139., 252.],
         [137., 259.]])
    return keypoints


def test_keypoints_DetectFaceKeypointNet2D32(
        image_with_faces,
        keypoints_DetectFaceKeypointNet2D32):
    cv2.ocl.setUseOpenCL(False)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(777)
    estimator = DetectFaceKeypointNet2D32()
    inferences = estimator(image_with_faces)
    predicted_keypoints = inferences['keypoints']
    assert len(predicted_keypoints) == len(keypoints_DetectFaceKeypointNet2D32)
    # TODO openCV is not deterministic with it's predictions
    print(predicted_keypoints)
    for label, preds in zip(
            keypoints_DetectFaceKeypointNet2D32, predicted_keypoints):
        assert np.allclose(label, preds)


def test_FaceKeypointNet2D32(image_with_face, keypoints_FaceKeypointNet2D32):
    estimator = FaceKeypointNet2D32()
    inferences = estimator(image_with_face)
    labelled_keypoints = keypoints_FaceKeypointNet2D32
    predicted_keypoints = inferences['keypoints']
    assert len(predicted_keypoints) == len(labelled_keypoints)
    assert np.allclose(predicted_keypoints, labelled_keypoints)
