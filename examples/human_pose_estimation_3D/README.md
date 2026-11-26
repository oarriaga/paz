# Human pose estimation 3D

Runs `paz.applications.EstimateHumanPose`, which returns 2D keypoints,
3D poses and a 6D pose. `viz.py` holds the 3D plotting helper.

## Run

```bash
python3 demo_image.py --image photo.jpg
python3 demo.py
python3 demo3D.py
```

- `demo_image.py`: options `--image` (required), `--HFOV` (default 70),
  `--output` (default `human_pose_6D.jpg`), `--output3D` (default
  `human_pose_3D.png`). Writes the drawn image and a 3D pose plot.
- `demo.py`: webcam only. Options `--camera` (0), `--H` (480), `--W`
  (640), `--HFOV` (70). Shows a video window.
- `demo3D.py`: webcam only, same options as `demo.py`. Shows the
  inference window and a live matplotlib 3D plot.

## Requirements

`scipy` and `matplotlib`. `demo.py` and `demo3D.py` need a webcam and
have no `--image` option.
