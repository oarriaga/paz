import cv2
import numpy as np

from paz.backend import video


def build_frames(num_frames=5, H=16, W=32, red=200):
    frames = np.zeros((num_frames, H, W, 3), dtype=np.uint8)
    frames[..., 0] = red
    return frames


def read_frames(filepath):
    capture = cv2.VideoCapture(str(filepath))
    frames = []
    while True:
        read, frame = capture.read()
        if not read:
            break
        frames.append(frame)
    capture.release()
    return frames


def test_from_frames_writes_every_frame(tmp_path):
    filepath = tmp_path / "video.mp4"
    video.from_frames(build_frames(), filepath, fps=10)
    assert len(read_frames(filepath)) == 5


def test_from_frames_writes_red_frames_as_BGR(tmp_path):
    filepath = tmp_path / "video.mp4"
    video.from_frames(build_frames(), filepath, fps=10)
    first = read_frames(filepath)[0]
    assert first[..., 2].mean() > first[..., 0].mean()
