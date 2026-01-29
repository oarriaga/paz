import os
import glob
import cv2


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
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(name, fourcc, fps, (W, H))
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
