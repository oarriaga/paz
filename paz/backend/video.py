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


def from_paths(image_paths, name="video.mp4", fps=10):
    H, W = imread(image_paths[0]).shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(name, fourcc, fps, (W, H))
    for image_path in image_paths:
        image = imread(image_path)
        image = (255.0 * image[..., :3][..., ::-1]).astype("uint8")
        video.write(image)
    video.release()


def from_directory(directory, wildcard="*.png", name="video.mp4", fps=10):
    image_wildcard = os.path.join(directory, wildcard)
    image_paths = sorted(glob.glob(image_wildcard))
    from_paths(image_paths, name, fps)
