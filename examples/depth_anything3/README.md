# Depth Anything 3

Depth estimation with Depth Anything 3. `demo.py` only fetches the demo
images used by the other scripts.

## Run

```bash
python3 estimate_single_view.py
python3 estimate_metric.py
python3 estimate_multi_view.py
```

Each script downloads its pretrained weights and demo image(s) when run
with no arguments.

- `estimate_single_view.py`: monocular relative depth (DA3MONO-LARGE).
  Options: `--weights` (default `pretrained`), `--image`, `--image_size`
  (518), `--output` (default `depth.png`). Writes a gray depth map.
- `estimate_metric.py`: metric depth in meters (DA3METRIC-LARGE).
  Options: `--weights`, `--image`, `--focal_length` (in pixels, default
  470.0), `--image_size`. Prints the median depth and the depth range.
- `estimate_multi_view.py`: depth and recovered cameras from several
  views (DA3-SMALL). Options: `--weights`, `--images`, `--image_size`.
  Prints the depth shape and the focal and translation of each view.
