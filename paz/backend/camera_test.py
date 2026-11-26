import cv2
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


class FakeCamera:
    def __init__(self, frame):
        self.frame = frame

    def start(self):
        pass

    def stop(self):
        pass

    def read(self):
        return self.frame


def write_video(path, frames_BGR):
    H, W = frames_BGR[0].shape[:2]
    fourCC = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourCC, 20, (W, H))
    for frame in frames_BGR:
        writer.write(frame)
    writer.release()


def read_video(path):
    video = cv2.VideoCapture(str(path))
    frames = []
    is_frame_received, frame = video.read()
    while is_frame_received:
        frames.append(frame)
        is_frame_received, frame = video.read()
    video.release()
    return np.array(frames)


def hide_windows(monkeypatch):
    monkeypatch.setattr(paz.image, "show", lambda *args, **kwargs: None)


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


def test_record_from_file_writes_pipeline_images(tmp_path, monkeypatch):
    hide_windows(monkeypatch)
    red_BGR = np.full((16, 32, 3), (0, 0, 255), np.uint8)
    write_video(tmp_path / "input.avi", [red_BGR] * 3)
    received_images = []

    def pipeline(image):
        received_images.append(image)
        return {"image": image}

    player = paz.VideoPlayer((8, 16), pipeline, None)
    output_path = str(tmp_path / "output.avi")
    player.record_from_file(str(tmp_path / "input.avi"), output_path)
    frames = read_video(output_path)
    assert frames.shape == (3, 8, 16, 3)
    assert np.allclose(frames, red_BGR[:8, :16], atol=10)
    assert np.allclose(received_images[0], red_BGR[..., ::-1], atol=10)


def test_record_frames_writes_camera_frames(tmp_path, monkeypatch):
    hide_windows(monkeypatch)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: ord("q"))
    red_RGB = np.full((16, 32, 3), (255, 0, 0), np.uint8)
    player = paz.VideoPlayer((8, 16), None, FakeCamera(red_RGB))
    output_path = str(tmp_path / "output.avi")
    player.record_frames(output_path)
    frames = read_video(output_path)
    assert frames.shape == (1, 8, 16, 3)
    assert np.allclose(frames, red_RGB[:8, :16, ::-1], atol=10)


def test_extract_frames_from_video_writes_selected_frames(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "images").mkdir()
    red_BGR = np.full((16, 32, 3), (0, 0, 255), np.uint8)
    write_video(tmp_path / "input.avi", [red_BGR] * 5)
    player = paz.VideoPlayer((8, 16), None, None)
    player.extract_frames_from_video(str(tmp_path / "input.avi"), 2)
    names = sorted(path.name for path in (tmp_path / "images").iterdir())
    assert names == ["0.jpg", "2.jpg", "4.jpg"]
    image = paz.image.load(tmp_path / "images" / "0.jpg")
    assert image.shape == (8, 16, 3)
    assert np.allclose(image, red_BGR[:8, :16, ::-1], atol=10)
