"""Track and count objects through a video with DeepSORT.

RF-DETR Nano finds the objects in every frame, a MobileNetV2 embedder
describes each box, and DeepSORT links them into tracks. Each track of a
counted class is counted once, after it was matched in ``min_track_frames``
frames; MaxN reports the most detections of a class seen in one frame. The
script writes the drawn video and a JSON with per-frame counts, track
records, final counts, MaxN and timing.

The detector uses the COCO weights for now; ``--weights`` and ``--names``
take a fine-tuned checkpoint once those load in paz.
"""
import argparse
import itertools
import json
import time

import cv2
import numpy as np

import paz
from paz.applications.detectors import RFDETR
from paz.models.tracking.deep_sort.application import DeepSORT
from paz.models.tracking.deep_sort.application import draw_track_box
from paz.models.tracking.deep_sort.tracker import CONFIRMED
from paz.models.tracking.deep_sort.tracker import Config
from paz.models.tracking.deep_sort.embedder import build_embedder
from paz.models.tracking.deep_sort.counting import select_classes
from paz.models.tracking.deep_sort.counting import build_counting
from paz.models.tracking.deep_sort.counting import count_frame_classes
from paz.models.tracking.deep_sort.counting import count_tracked_frames
from paz.models.tracking.deep_sort.counting import build_frame_record
from paz.models.tracking.deep_sort.counting import build_summary


def build_detector(weights, names, score_thresh):
    if weights is None:
        detect = paz.applications.DetectRFDETRNano(score_thresh, draw=False)
        names = ["0"] + paz.datasets.labels("COCO_EFFICIENTDET")
    else:
        names = names.split(",")
        detect = build_fine_tuned_detector(weights, names, score_thresh)
    return detect, names


def build_fine_tuned_detector(weights, names, score_thresh):
    model = paz.models.RFDETRNano(len(names))
    model.load_weights(weights)
    return RFDETR(model, score_thresh, False)


def build_detect(detect, names, count_classes):
    def detect_names(image):
        boxes2D, labels, scores = detect(image)
        class_names = [names[int(label)] for label in labels]
        return select_classes(boxes2D, class_names, scores, count_classes)

    return detect_names


def open_video(path, start_frame):
    capture = cv2.VideoCapture(path)
    if start_frame > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    return capture


def read_fps(capture):
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    return fps if np.isfinite(fps) and fps > 0 else 25.0


def read_frames(capture, max_frames):
    return itertools.islice(read_all_frames(capture), max_frames)


def read_all_frames(capture):
    received, frame = capture.read()
    while received:
        yield paz.image.BGR_to_RGB(frame)
        received, frame = capture.read()


def pair_frames(track, frames):
    # frames may be a one-shot generator, read by the tracker and the pairs
    images, inputs = itertools.tee(frames)
    return zip(images, track(inputs))


def time_steps(steps):
    start = time.perf_counter()
    for step in steps:
        yield step, 1000.0 * (time.perf_counter() - start)
        start = time.perf_counter()


def draw_tracks(image, tracks, display_ids, max_missed):
    image = np.array(image, np.uint8)
    for track in tracks:
        if is_drawn(track, max_missed):
            label = label_track(track, display_ids)
            image = draw_track_box(image, track, label)
    return image


def is_drawn(track, max_missed):
    is_recent = track.time_since_update <= max_missed
    return track.state == CONFIRMED and is_recent


def label_track(track, display_ids):
    if track.id in display_ids:
        label = f"{track.class_name} ID {display_ids[track.id]}"
    else:
        label = track.class_name
    return label


def draw_counts(image, counts):
    image = np.array(image, np.uint8)
    for line_arg, (name, count) in enumerate(counts.items()):
        point = (10, 30 + 30 * line_arg)
        image = paz.draw.text(image, f"{name}: {count}", point)
    return image


def write_json(path, payload):
    with open(path, "w") as json_file:
        json.dump(payload, json_file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="deep_sort_track.mp4")
    parser.add_argument("--json_output", default="deep_sort_track.json")
    parser.add_argument("--score_thresh", type=float, default=0.3)
    parser.add_argument("--count_classes", default="person")
    parser.add_argument("--min_track_frames", type=int, default=5)
    parser.add_argument("--max_draw_missed_frames", type=int, default=5)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--names", default=None)
    arguments = parser.parse_args()
    if arguments.weights is not None and arguments.names is None:
        parser.error("--weights needs --names, the class names it predicts")

    count_classes = arguments.count_classes.split(",")
    args = (arguments.weights, arguments.names, arguments.score_thresh)
    detect, names = build_detector(*args)
    detect = build_detect(detect, names, count_classes)
    embed = build_embedder(paz.models.MobileNetV2())
    config = Config()
    track = DeepSORT(detect, embed, config, False)

    capture = open_video(arguments.video, arguments.start_frame)
    fps = read_fps(capture)
    W = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_input_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = paz.video.build_writer(arguments.output, fps, H, W)

    frames = read_frames(capture, arguments.max_frames)
    args = (count_classes, arguments.min_track_frames)
    steps = time_steps(count_tracked_frames(pair_frames(track, frames), *args))
    counting = build_counting(count_classes)
    max_counts = count_frame_classes([], count_classes)
    records, times_ms = [], []
    for frame_index, step in enumerate(steps, arguments.start_frame):
        counted, time_ms = step
        counting, max_counts = counted.counting, counted.max_counts
        args = (counting.display_ids, arguments.max_draw_missed_frames)
        image = draw_tracks(counted.image, counted.tracks, *args)
        image = draw_counts(image, counting.counts)
        writer.write(paz.to_numpy(paz.image.RGB_to_BGR(image)))
        records.append(build_frame_record(frame_index, time_ms, counted))
        times_ms.append(time_ms)
    capture.release()
    writer.release()

    payload = dict(input=arguments.video, weights=arguments.weights)
    payload.update(tracked_video=arguments.output, class_names=names)
    payload.update(count_classes=count_classes)
    payload.update(threshold=arguments.score_thresh)
    payload.update(min_track_frames=arguments.min_track_frames)
    deep_sort = config._asdict()
    deep_sort.update(max_draw_missed_frames=arguments.max_draw_missed_frames)
    deep_sort.update(embedder="paz MobileNetV2")
    payload.update(deepsort=deep_sort, fps=fps, width=W, height=H)
    payload.update(input_frame_count=num_input_frames)
    payload.update(start_frame=arguments.start_frame)
    payload.update(max_frames=arguments.max_frames, frames=records)
    payload.update(build_summary(counting, max_counts, times_ms))
    write_json(arguments.json_output, payload)
    print("final counts", counting.counts, "MaxN", max_counts)
    print("saved", arguments.output, "and", arguments.json_output)
