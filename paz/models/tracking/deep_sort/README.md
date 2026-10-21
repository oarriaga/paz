# Track and count objects with DeepSORT

`track_video.py` reads a video, detects objects in every frame with
RF-DETR Nano, describes each box with a MobileNetV2 embedder, links the boxes
into tracks with DeepSORT (`paz.models.tracking.deep_sort.DeepSORT`), and
counts every track once. It writes the drawn video and a JSON report.

```bash
KERAS_BACKEND=jax python -m paz.models.tracking.deep_sort.track_video \
    --video clip.mp4 --count_classes person --max_frames 200
```

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--video` | required | input video |
| `--output` | `deep_sort_track.mp4` | drawn video |
| `--json_output` | `deep_sort_track.json` | report |
| `--score_thresh` | `0.3` | detector score threshold |
| `--count_classes` | `person` | comma-separated classes to track and count |
| `--min_track_frames` | `5` | matched frames needed before a track counts |
| `--max_draw_missed_frames` | `5` | most missed frames a drawn track has |
| `--start_frame` | `0` | first frame to read |
| `--max_frames` | all | number of frames to read |
| `--weights` | COCO | fine-tuned RF-DETR Nano weights |
| `--names` | COCO | comma-separated class names of `--weights` |

The tracker uses `Config()` defaults: `max_age=30`, `min_hits=3`,
`max_IOU_distance=0.7`, `max_cosine_distance=0.2`.

## Counting rule

- Detections of other classes are dropped **before** tracking. Class names
  are compared case-insensitively and counted under the spelling given in
  `--count_classes`.
- A track's length is the number of frames it was matched in
  (`time_since_update == 0`), tentative frames included.
- A track is counted once, on the first frame where it is confirmed, matched
  in that frame, of a counted class, and at least `--min_track_frames` long.
  It is counted under its class in that frame.
- Counted tracks get a display id per class, in track order.
- MaxN is the largest number of detections of a class in one frame.

## Outputs

- The video shows confirmed tracks missed at most `--max_draw_missed_frames`
  frames, labelled with their class and display id, plus the running counts.
- The JSON holds the run settings, then one record per frame
  (`frame_index`, `pipeline_inference_time_ms`, `num_detections`,
  `num_tracks`, `counts`, `newly_counted_track_ids`, `detections`, `tracks`,
  `maxn`), then `processed_frames`, `final_counts`, `counted_track_ids`
  (sorted as strings), `counted_display_ids`, `track_class_names`,
  `final_maxn` and `pipeline_timing` (mean, median, p95, FPS and the first,
  compiling frame).

## Differences from the reference script

The reference is Jichen's `track_count_deepsort_paper.py`, whose
tracking is deep-sort-realtime 1.3.2. See [NOTICE.md][notice] for the
weights and licences.

[notice]: mobilenet_v2/NOTICE.md

- The detector uses the COCO weights. Fine-tuned fish weights follow once
  upstream #438 lets them load; until then `--weights` and `--names` are
  untested.
- paz detectors return int32 boxes clipped to the frame; DeepSORT receives
  them as float32 `[x, y, w, h]`.
- The MobileNetV2 weights are ported from deep-sort-realtime 1.3.2's
  MobileNetV2 embedder and hosted on a fork release until they are
  mirrored.
- `num_detections`, `detections` and MaxN count the detections kept after
  the class filter and the box-size check. Detection records carry no
  integer `label`.
- The frame time includes reading the frame and excludes drawing and
  writing.
- Boxes and text are drawn with `paz.draw`.

## Tests

```bash
python -m pytest paz/models/tracking -q
```

The parity test repeats the synthetic run on deep-sort-realtime 1.3.2 and
compares the counts frame by frame. It needs that package installed:

```bash
PAZ_DEEP_SORT_PARITY=1 python -m pytest paz/models/tracking -rs
```
