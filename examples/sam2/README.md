# SAM 2

`segment_image.py` interactively segments multiple objects in an image
with SAM 2.1. `track_video.py` tracks objects through a video and writes
the overlay video. `prompt_selector.py` holds the prompt selector used
by `segment_image.py`.

## Run

```bash
python3 segment_image.py
python3 track_video.py
```

`segment_image.py` options: `--image` (unset downloads a demo image),
`--output` (default `sam2_mask.png`). It is interactive.

`track_video.py` options: `--video` (unset downloads a demo video),
`--point` (two integers), `--box` (four integers), `--num_frames` (60),
`--stride` (3), `--output` (default `sam2_track.mp4`). The point and the
box prompt one object each on the first frame.

## Output

`segment_image.py` writes the result to `--output`. `track_video.py`
prints the pixels covered per frame and writes the overlay video to
`--output`.
