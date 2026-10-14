import os
import glob

import cv2

from paz.backend.image import RGB_to_BGR
from paz.backend.standard import to_numpy


def from_directories(
    directories, video_name="video.mp4", wildcard="*.png", fps=10
):
    total_image_files = []
    for directory in directories:
        image_wildcard = os.path.join(directory, wildcard)
        image_files = sorted(glob.glob(image_wildcard))
        total_image_files.extend(image_files)
    if not total_image_files:
        raise ValueError("No images found for video.")
    from_paths(total_image_files, video_name, fps)


def from_paths(image_paths, name="video.mp4", fps=10):
    if not image_paths:
        raise ValueError("No images found for video.")
    first_image = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)
    if first_image is None:
        raise ValueError(f"Failed to read image: {image_paths[0]}")
    H, W = first_image.shape[:2]
    video = build_writer(name, fps, H, W)
    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        video.write(image)
    video.release()


def from_directory(directory, wildcard="*.png", name="video.mp4", fps=10):
    image_wildcard = os.path.join(directory, wildcard)
    image_paths = sorted(glob.glob(image_wildcard))
    from_paths(image_paths, name, fps)


def from_frames(frames, name="video.mp4", fps=10):
    if len(frames) == 0:
        raise ValueError("No frames found for video.")
    H, W = frames[0].shape[:2]
    video = build_writer(name, fps, H, W)
    for frame in frames:
        # frames are held in paz's RGB convention; cv2 writes BGR
        video.write(to_numpy(RGB_to_BGR(frame)))
    video.release()


def build_writer(name, fps, H, W):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(name), fourcc, fps, (W, H))
