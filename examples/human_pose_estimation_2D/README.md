# Human pose estimation 2D

Runs `paz.applications.HigherHRNetHumanPose2D` on an image or on a webcam
stream.

## Run

```bash
python3 demo.py --image photo.jpg
python3 demo.py
```

Options: `--image` (RGB image; webcam if unset), `--camera` (default 0),
`--H` (480), `--W` (640), `--output` (default `human_pose_2D.jpg`).

## Output

With `--image` the drawn image is written to `--output`. Without it a
video window shows the webcam stream; this needs a webcam.
