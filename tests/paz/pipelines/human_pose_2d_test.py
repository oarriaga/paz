import pytest
import os
import numpy as np
from tensorflow.keras.utils import get_file
from paz.backend.image import load_image
from paz.applications import HigherHRNetHumanPose2D


@pytest.fixture
def image_with_single_person():
    URL = ('https://github.com/oarriaga/altamira-data/releases/download'
           '/v0.10/single_person_test_pose.png')
    filename = os.path.basename(URL)
    fullpath = get_file(filename, URL, cache_subdir='paz/tests')
    image = load_image(fullpath)
    return image


@pytest.fixture
def image_with_multi_person():
    URL = ('https://github.com/oarriaga/altamira-data/releases/download'
           '/v0.10/multi_person_test_pose.png')
    filename = os.path.basename(URL)
    fullpath = get_file(filename, URL, cache_subdir='paz/tests')
    image = load_image(fullpath)
    return image


@pytest.fixture
def labeled_joint_single_person():
    return np.array([[197.07764, 69.019040],
                     [205.26025, 61.615723],
                     [188.50537, 62.005370],
                     [217.33936, 67.850100],
                     [176.42627, 68.629395],
                     [246.56299, 121.62158],
                     [153.82666, 114.99756],
                     [264.48682, 187.47217],
                     [109.79639, 144.61084],
                     [247.73193, 257.99854],
                     [117.19971, 84.994630],
                     [229.02881, 268.12940],
                     [174.08838, 268.12940],
                     [224.35303, 360.47607],
                     [180.32275, 362.42432],
                     [229.02881, 457.49854],
                     [177.98486, 455.93994]])


@pytest.fixture
def labeled_scores_single_person():
    return [1.1975062]


@pytest.fixture
def labeled_joint_multi_person():
    return [np.array([[399.73584, 698.07960],
                      [425.06300, 678.08450],
                      [370.40967, 671.41943],
                      [453.05615, 706.07764],
                      [326.42040, 695.41360],
                      [475.71730, 832.71340],
                      [261.10303, 850.04250],
                      [495.71240, 1045.9946],
                      [218.44678, 1093.9829],
                      [478.38330, 1221.9517],
                      [299.76025, 1289.9350],
                      [415.73193, 1215.2866],
                      [291.76220, 1212.6206],
                      [361.07860, 1488.5532],
                      [275.76610, 1495.2183],
                      [361.07860, 1711.1655],
                      [222.44580, 1769.8179]]),
            np.array([[603.68604, 562.11280],
                      [633.01220, 539.45166],
                      [581.02490, 540.78467],
                      [678.33450, 574.10986],
                      [550.36570, 567.44480],
                      [746.31790, 743.40186],
                      [479.71630, 716.74170],
                      [799.63820, 947.35205],
                      [330.41943, 780.72610],
                      [726.32275, 1129.9741],
                      [190.45361, 828.71436],
                      [671.66943, 1141.9712],
                      [513.04150, 1128.6411],
                      [666.33740, 1415.2378],
                      [539.70166, 1416.5708],
                      [669.00340, 1643.1821],
                      [545.03370, 1659.1782]]),
            np.array([[399.73584, 698.07960],
                      [425.06300, 678.08450],
                      [370.40967, 671.41943],
                      [453.05615, 706.07764],
                      [326.42040, 695.41360],
                      [475.71730, 832.71340],
                      [261.10303, 850.04250],
                      [495.71240, 1045.9946],
                      [218.44678, 1093.9829],
                      [478.38330, 1221.9517],
                      [299.76025, 1289.9350],
                      [415.73193, 1215.2866],
                      [291.76220, 1212.6206],
                      [361.07860, 1488.5532],
                      [277.09912, 1484.5542],
                      [361.07860, 1711.1655],
                      [222.44580, 1769.8179]])]


@pytest.fixture
def labeled_scores_multi_person():
    return [0.8910445, 1.1566495, 0.07124636]


@pytest.fixture
def dataset():
    return 'COCO'


@pytest.fixture
def data_with_center():
    return False


@pytest.fixture
def joint_order():
    return [0, 1, 2, 3, 4, 5, 6, 11, 12, 7, 8, 9, 10, 13, 14, 15, 16]


@pytest.fixture
def flipped_joint_order():
    return [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def test_DetectHumanPose2D_single_person(image_with_single_person,
                                         labeled_joint_single_person,
                                         labeled_scores_single_person):
    detect = HigherHRNetHumanPose2D()
    inferences = detect(image_with_single_person)
    assert np.allclose(inferences['scores'], labeled_scores_single_person)
    assert np.allclose(inferences['keypoints'], labeled_joint_single_person)


def test_DetectHumanPose2D_multi_person(image_with_multi_person,
                                        labeled_joint_multi_person,
                                        labeled_scores_multi_person):
    detect = HigherHRNetHumanPose2D()
    inferences = detect(image_with_multi_person)
    assert np.allclose(inferences['scores'], labeled_scores_multi_person)
    assert np.allclose(inferences['keypoints'], labeled_joint_multi_person)
