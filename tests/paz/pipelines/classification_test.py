import os
import pytest
import numpy as np

from tensorflow.keras.utils import get_file
from paz.backend.image import load_image
from paz.pipelines.classification import MiniXceptionFER


@pytest.fixture
def image_with_face():
    URL = ('https://github.com/oarriaga/altamira-data/releases/download'
           '/v0.9.1/image_with_face.jpg')
    filename = os.path.basename(URL)
    fullpath = get_file(filename, URL, cache_subdir='paz/tests')
    image = load_image(fullpath)
    return image


@pytest.fixture
def labeled_scores():
    return np.array([[6.9692191e-03, 6.5534514e-05, 3.6219540e-03,
                      8.2652807e-01, 4.4210157e-03, 1.0055617e-03,
                      1.5738861e-01]])


@pytest.fixture
def labeled_emotion():
    return 'happy'


def test_MiniXceptionFER(image_with_face, labeled_emotion, labeled_scores):
    classifier = MiniXceptionFER()
    inferences = classifier(image_with_face)
    assert inferences['class_name'] == labeled_emotion
    assert np.allclose(inferences['scores'], labeled_scores)
