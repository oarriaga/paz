# Head pose estimation 6D

Runs `paz.applications.HeadPoseKeypointNet2D32` on a webcam stream. The
camera intrinsics are computed from the horizontal field of view.

## Run

```bash
python3 demo.py --camera 0
```

Options: `--camera` (default 4), `--box_scale` (default 1.2), `--H`
(480), `--W` (640).

## Output

A video window with the webcam stream and the pipeline output.

## Requirements

A webcam; the script has no `--image` option. The default camera
identifier is 4, so pass `--camera` for your device.
