import pytest
import os
import numpy as np
from tensorflow.keras.utils import get_file
from paz.backend.image import load_image
from paz.applications import MinimalHandPoseEstimation


@pytest.fixture
def image():
    URL = ('https://github.com/oarriaga/altamira-data/releases/download'
           '/v0.14/image_with_hand.png')
    filename = os.path.basename(URL)
    fullpath = get_file(filename, URL, cache_subdir='paz/tests')
    image = load_image(fullpath)
    return image


@pytest.fixture
def keypoints3D():
    return np.array([[4.59543616e-02, 1.01974916e+00, -4.18092608e-02],
                     [1.97005451e-01, 7.27735639e-01, 9.93182510e-03],
                     [4.65905547e-01, 4.04410481e-01, 2.02238679e-01],
                     [5.27417541e-01, 1.19056322e-01, 3.12423348e-01],
                     [7.06151009e-01, -1.45121396e-01, 5.78676224e-01],
                     [2.29134336e-01, 5.17019629e-03, -1.86777227e-02],
                     [1.90560371e-01, -4.16686863e-01, 3.08839791e-02],
                     [2.45253161e-01, -6.03881776e-01, 1.07773989e-02],
                     [2.45344773e-01, -9.46746171e-01, 3.10686044e-03],
                     [-1.17659438e-10, 6.89383772e-10, -1.66820258e-09],
                     [-1.06274009e-01, -3.72214854e-01, -1.83141753e-02],
                     [-1.03316717e-01, -7.05676019e-01, 5.39230146e-02],
                     [-1.03476852e-01, -9.64827240e-01, 4.52297479e-02],
                     [-1.65308565e-01, 2.90882755e-02, 7.35934004e-02],
                     [-2.57835418e-01, -2.79438645e-01, 5.13521582e-02],
                     [-3.15736741e-01, -5.79943478e-01, 7.48071373e-02],
                     [-3.92356992e-01, -8.97268593e-01, 2.25844055e-01],
                     [-3.08868229e-01, 1.16529495e-01, 1.68661669e-01],
                     [-4.35660630e-01, -1.56533003e-01, 2.14194521e-01],
                     [-5.57054341e-01, -3.33916962e-01, 2.07406536e-01],
                     [-6.17552280e-01, -5.51497638e-01, 3.22265536e-01]])


@pytest.fixture
def keypoints2D():
    return [np.array([[300, 381],
                      [356, 337],
                      [393, 293],
                      [412, 235],
                      [450, 205],
                      [337, 205],
                      [356, 132],
                      [356, 73],
                      [356, 29],
                      [300, 205],
                      [300, 132],
                      [300, 73],
                      [281, 14],
                      [262, 205],
                      [243, 146],
                      [225, 88],
                      [225, 44],
                      [225, 220],
                      [206, 176],
                      [187, 132],
                      [168, 102]])]


def test_MinimalHandPoseEstimation(image, keypoints3D, keypoints2D):
    detect = MinimalHandPoseEstimation()
    inferences = detect(image)
    assert np.allclose(inferences['keypoints3D'], keypoints3D, rtol=1e-03)
    assert np.allclose(inferences['keypoints2D'], keypoints2D, rtol=1e-03)
