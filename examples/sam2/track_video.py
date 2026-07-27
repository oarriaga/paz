"""Track objects through a video with SAM 2.1.

Point at each object on the first frame and SAM 2 follows it through the rest
of the video, writing one overlay image per frame. Without ``--video`` the
example pans across a photo, which needs no download beyond the photo itself
and still exercises the full memory path.
"""
import os
import argparse

import numpy as np
import cv2
from keras.utils import get_file

import paz
from paz.models.foundation.sam2.video import Prompt

SHIFT = 14


def fetch_image():
    URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
    path = get_file("sam2_cats.jpg", URL, cache_subdir="paz/examples/sam2")
    return paz.image.load(path)


def build_panning_frames(num_frames=8):
    image = paz.image.resize(fetch_image(), (768, 768), "linear", True)
    pixels = np.asarray(paz.to_numpy(image), np.uint8)
    shifts = [index * SHIFT for index in range(num_frames)]
    return [np.roll(pixels, shift, axis=1) for shift in shifts]


def load_frames(path):
    if os.path.isdir(path):
        names = sorted(os.listdir(path))
        frames = [paz.image.load(os.path.join(path, name)) for name in names]
    else:
        frames = read_video(path)
    return [np.asarray(paz.to_numpy(frame), np.uint8) for frame in frames]


def read_video(path):
    video = cv2.VideoCapture(path)
    frames = []
    while True:
        received, frame = video.read()
        if not received:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    video.release()
    return frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None)
    parser.add_argument("--point", type=int, nargs=2, default=None)
    parser.add_argument("--output", default="sam2_track")
    arguments = parser.parse_args()

    if arguments.video is None:
        frames = build_panning_frames()
    else:
        frames = load_frames(arguments.video)
    height, width = frames[0].shape[:2]
    point = arguments.point or (width // 2, int(height * 0.6))
    prompts = [Prompt(0, 1, points=[point], labels=[1])]
    print("tracking", len(frames), "frames from", point)

    track = paz.applications.TrackSAMHieraSmall21()
    os.makedirs(arguments.output, exist_ok=True)
    for frame, masks, overlay in track(frames, prompts):
        path = os.path.join(arguments.output, f"{frame:05d}.png")
        paz.image.write(path, overlay)
        print("frame", frame, "covered", int(masks.sum()), "pixels")
    print("saved overlays to", arguments.output)
