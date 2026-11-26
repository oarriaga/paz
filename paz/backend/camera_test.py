import numpy as np

import paz


class FakeCapture:
    def __init__(self, frames):
        self.frames = list(frames)

    def read(self):
        if self.frames:
            output = (True, self.frames.pop(0))
        else:
            output = (False, None)
        return output


def test_camera_read_returns_every_frame_once():
    blue_BGR = np.full((2, 2, 3), (255, 0, 0), np.uint8)
    red_BGR = np.full((2, 2, 3), (0, 0, 255), np.uint8)
    camera = paz.Camera()
    camera._camera = FakeCapture([blue_BGR, red_BGR])
    assert np.array_equal(camera.read(), blue_BGR[..., ::-1])
    assert np.array_equal(camera.read(), red_BGR[..., ::-1])
    assert camera.read() is None


def test_camera_keeps_distortion():
    distortion = np.ones(5)
    camera = paz.Camera(distortion=distortion)
    assert camera.distortion is distortion
