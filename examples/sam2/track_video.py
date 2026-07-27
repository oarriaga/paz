"""Track objects through a video with SAM 2.1 and write the overlay video.

Point at each object on the first frame and SAM 2 follows it through the rest of
the video. Every sub-model is jitted and the memory bank is padded to a fixed
size, so each one compiles once and the tracker then runs at a steady cost:
about 0.2 s and 1.3 GB of GPU memory per frame for the small backbone.

Those first-run compiles cost about 30 s. To keep them across runs, export
``JAX_COMPILATION_CACHE_DIR``, ``JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1``
and ``JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0`` before running; JAX reads
them while it initializes, so setting them from Python here would be too late.

The default video and prompts are the two children of the official SAM 2 demo
clip, downloaded on first use.
"""
import argparse

import cv2
from keras.utils import get_file

import paz
from paz.models.foundation.sam2.video import Prompt

SAM2_REPO = "https://raw.githubusercontent.com/facebookresearch/sam2/main/"
DEMO_VIDEO = SAM2_REPO + "notebooks/videos/bedroom.mp4"
BOY = (250, 220)
GIRL = (322, 0, 522, 392)


def fetch_video():
    kwargs = dict(cache_subdir="paz/examples/sam2")
    return get_file("sam2_bedroom.mp4", DEMO_VIDEO, **kwargs)


def read_frames(path, num_frames, stride):
    capture = cv2.VideoCapture(path)
    frames = []
    while len(frames) < num_frames:
        received, frame = capture.read()
        if not received:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for _ in range(stride - 1):
            capture.read()
    capture.release()
    return frames


def write_video(path, frames, fps):
    height, width = frames[0].shape[:2]
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, codec, fps, (width, height))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None)
    parser.add_argument("--point", type=int, nargs=2, default=BOY)
    parser.add_argument("--box", type=int, nargs=4, default=GIRL)
    parser.add_argument("--num_frames", type=int, default=60)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--output", default="sam2_track.mp4")
    arguments = parser.parse_args()

    path = arguments.video or fetch_video()
    frames = read_frames(path, arguments.num_frames, arguments.stride)
    print("read", len(frames), "frames of", frames[0].shape)
    clicked = Prompt(0, 1, points=[arguments.point], labels=[1])
    prompts = [clicked, Prompt(0, 2, box=arguments.box)]

    track = paz.applications.TrackSAMHieraSmall21()
    overlays = []
    for frame, masks, overlay in track(frames, prompts):
        overlays.append(overlay)
        print("frame", frame, "covered", int(masks.sum()), "pixels")
    write_video(arguments.output, overlays, 30.0 / arguments.stride)
    print("saved", arguments.output)
