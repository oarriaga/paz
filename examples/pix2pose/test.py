from paz.abstract import SequentialProcessor, Processor
from paz import processors as pr
import numpy as np
from backend import build_cube_points3D
# import pytest


class PipelineWithTwoChannels(SequentialProcessor):
    def __init__(self):
        super(PipelineWithTwoChannels, self).__init__()
        self.add(lambda x: x)
        self.add(pr.ControlMap(pr.Copy(), [0], [1], keep={0: 0}))


class PipelineWithThreeChannels(SequentialProcessor):
    def __init__(self):
        super(PipelineWithThreeChannels, self).__init__()
        self.add(lambda a, b: (a, b))
        self.add(pr.ControlMap(pr.Copy(), [0], [2], keep={0: 0}))


class PipelineWithThreeChannelsPlus(SequentialProcessor):
    def __init__(self):
        super(PipelineWithThreeChannelsPlus, self).__init__()
        self.add(lambda a, b: (a, b))
        self.add(pr.ControlMap(pr.Copy(), [0], [2], keep={0: 0}))
        self.add(pr.ControlMap(SumTwoValues(), [0, 1], [0]))


class SumTwoValues(Processor):
    def __init__(self):
        super(SumTwoValues, self).__init__()

    def call(self, A, B):
        return A + B


def test_copy_with_controlmap_using_2_channels():
    pipeline = PipelineWithTwoChannels()
    random_values = np.random.random((128, 128))
    values = pipeline(random_values)
    assert len(values) == 2
    assert np.allclose(values[0], random_values)
    assert np.allclose(values[1], random_values)


def test_copy_with_controlmap_using_3_channels():
    pipeline = PipelineWithThreeChannels()
    A_random_values = np.random.random((128, 128))
    B_random_values = np.random.random((128, 128))
    values = pipeline(A_random_values, B_random_values)
    assert len(values) == 3
    assert np.allclose(values[0], A_random_values)
    assert np.allclose(values[1], B_random_values)
    assert np.allclose(values[2], A_random_values)


def test_copy_with_controlmap_using_3_channels_plus():
    pipeline = PipelineWithThreeChannelsPlus()
    A_random_values = np.random.random((128, 128))
    B_random_values = np.random.random((128, 128))
    values = pipeline(A_random_values, B_random_values)
    assert len(values) == 2
    assert np.allclose(values[0], A_random_values + B_random_values)
    assert np.allclose(values[1], A_random_values)


def test_build_cube_points3D(width, height, depth):
    cube_points3D = build_cube_points3D(width, height, depth)
