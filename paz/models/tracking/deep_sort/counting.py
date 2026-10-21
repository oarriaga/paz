"""Count-once and MaxN rules for tracked videos.

A track is counted once, on the first frame where it is confirmed, matched
in that frame, of a counted class, and matched in at least
``min_track_frames`` frames so far (tentative frames included). Classes are
compared case-insensitively and counted under the spelling given by the
user. MaxN is the largest number of detections of a class in one frame.
"""
from collections import namedtuple

import numpy as np

from paz.models.tracking.deep_sort.tracker import CONFIRMED
from paz.models.tracking.deep_sort.tracker import compute_box2D

Counting = namedtuple("Counting", "counts lengths display_ids class_names")
FRAME_FIELDS = "image detections tracks counting newly_counted_ids max_counts"
CountedFrame = namedtuple("CountedFrame", FRAME_FIELDS)
RATE_NAMES = (
    "mean_ms_per_frame", "median_ms_per_frame", "p95_ms_per_frame",
    "pipeline_fps", "first_frame_ms",
)


def select_classes(boxes2D, class_names, scores, count_classes):
    kept_args = np.array(find_counted_args(class_names, count_classes), int)
    kept_names = [class_names[arg] for arg in kept_args]
    boxes2D = np.asarray(boxes2D).reshape(-1, 4)[kept_args]
    return boxes2D, kept_names, np.asarray(scores)[kept_args]


def find_counted_args(class_names, count_classes):
    lowered = [count_class.lower() for count_class in count_classes]
    kept_args = []
    for arg, class_name in enumerate(class_names):
        if class_name.lower() in lowered:
            kept_args.append(arg)
    return kept_args


def build_counting(count_classes):
    counts = {count_class: 0 for count_class in count_classes}
    return Counting(counts, {}, {}, {})


def count_tracked_frames(tracked_frames, count_classes, min_track_frames):
    counting = build_counting(count_classes)
    max_counts = count_frame_classes([], count_classes)
    for image, tracked in tracked_frames:
        args = (counting, tracked.tracks, min_track_frames)
        counting, newly_counted_ids = update_counting(*args)
        max_counts = add_frame_counts(max_counts, tracked.detections)
        args = (image, tracked.detections, tracked.tracks, counting)
        yield CountedFrame(*args, newly_counted_ids, max_counts)


def add_frame_counts(max_counts, detections):
    class_names = [detection.class_name for detection in detections]
    frame_counts = count_frame_classes(class_names, list(max_counts))
    return update_max_counts(max_counts, frame_counts)


def update_counting(counting, tracks, min_track_frames):
    counting = record_matches(counting, tracks)
    newly_counted_ids = []
    for track in tracks:
        if is_countable(counting, track, min_track_frames):
            counting = count_track(counting, track)
            newly_counted_ids.append(track.id)
    return counting, newly_counted_ids


def record_matches(counting, tracks):
    lengths, class_names = dict(counting.lengths), dict(counting.class_names)
    for track in tracks:
        if track.time_since_update == 0:
            lengths[track.id] = lengths.get(track.id, 0) + 1
            class_names[track.id] = track.class_name
    return counting._replace(lengths=lengths, class_names=class_names)


def is_countable(counting, track, min_track_frames):
    is_matched = track.state == CONFIRMED and track.time_since_update == 0
    is_new = track.id not in counting.display_ids
    is_long = counting.lengths.get(track.id, 0) >= min_track_frames
    count_classes = find_count_classes(counting.counts, track.class_name)
    return is_matched and is_new and is_long and len(count_classes) > 0


def count_track(counting, track):
    class_name = find_count_classes(counting.counts, track.class_name)[0]
    counts = dict(counting.counts)
    counts[class_name] = counts[class_name] + 1
    display_ids = {**counting.display_ids, track.id: counts[class_name]}
    return counting._replace(counts=counts, display_ids=display_ids)


def find_count_classes(count_classes, class_name):
    found = []
    for count_class in count_classes:
        if count_class.lower() == class_name.lower():
            found.append(count_class)
    return found


def count_frame_classes(class_names, count_classes):
    lowered = [class_name.lower() for class_name in class_names]
    counts = {}
    for count_class in count_classes:
        counts[count_class] = lowered.count(count_class.lower())
    return counts


def update_max_counts(max_counts, frame_counts):
    updated = {}
    for class_name, count in max_counts.items():
        updated[class_name] = max(count, frame_counts[class_name])
    return updated


def build_frame_record(frame_index, time_ms, counted_frame):
    record = dict(frame_index=frame_index)
    record.update(pipeline_inference_time_ms=float(time_ms))
    record.update(describe_frame(counted_frame))
    record.update(maxn=dict(counted_frame.max_counts))
    return record


def describe_frame(counted_frame):
    detections = to_detection_records(counted_frame.detections)
    tracks = to_track_records(counted_frame.tracks, counted_frame.counting)
    record = dict(num_detections=len(detections), num_tracks=len(tracks))
    record.update(counts=dict(counted_frame.counting.counts))
    newly_counted_ids = list(counted_frame.newly_counted_ids)
    record.update(newly_counted_track_ids=newly_counted_ids)
    return dict(record, detections=detections, tracks=tracks)


def to_detection_records(detections):
    return [to_detection_record(detection) for detection in detections]


def to_detection_record(detection):
    record = dict(box_xyxy=to_corners(detection.box2D))
    record.update(score=float(detection.score))
    return dict(record, class_name=detection.class_name)


def to_track_records(tracks, counting):
    records = []
    for track in tracks:
        if track.state == CONFIRMED:
            records.append(to_track_record(track, counting))
    return records


def to_track_record(track, counting):
    display_id = counting.display_ids.get(track.id)
    record = dict(track_id=track.id, display_id=display_id)
    record.update(class_name=track.class_name)
    record.update(box_xyxy=to_corners(compute_box2D(track)))
    record.update(length_frames=counting.lengths.get(track.id, 0))
    record.update(time_since_update=int(track.time_since_update))
    return dict(record, counted=track.id in counting.display_ids)


def to_corners(box2D):
    x_min, y_min, W, H = [float(value) for value in box2D]
    return [x_min, y_min, x_min + W, y_min + H]


def build_summary(counting, max_counts, times_ms):
    summary = dict(processed_frames=len(times_ms))
    summary.update(final_counts=dict(counting.counts))
    summary.update(counted_track_ids=sorted(counting.display_ids))
    summary.update(counted_display_ids=dict(counting.display_ids))
    summary.update(track_class_names=dict(counting.class_names))
    summary.update(final_maxn=dict(max_counts))
    return dict(summary, pipeline_timing=summarize_timing(times_ms))


def summarize_timing(times_ms):
    total_seconds = float(np.sum(times_ms)) / 1000.0
    if len(times_ms) == 0:
        rates = dict.fromkeys(RATE_NAMES)
    else:
        rates = compute_rates(times_ms, total_seconds)
    scope = "per frame: decode, detect, embed, track and count"
    return dict(scope=scope, total_seconds=total_seconds, **rates)


def compute_rates(times_ms, total_seconds):
    values = [np.mean(times_ms), np.median(times_ms)]
    values = values + [np.percentile(times_ms, 95)]
    values = values + [len(times_ms) / total_seconds, times_ms[0]]
    return dict(zip(RATE_NAMES, [float(value) for value in values]))
