# Minimal hand pose estimation

Hand keypoint detection (DetNet) and joint-angle estimation (IKNet), optionally
front-ended by an SSD512 hand detector for full-image, multi-hand inference.

All scripts default to `KERAS_BACKEND=jax`.

## Scripts

- `demo_image.py` — run `MinimalHandPoseEstimation` on a single image
  (downloads a test image if `--image` is unset) and write the drawn result.
- `demo.py` — real-time `MinimalHandPoseEstimation` from a webcam.
- `detect_demo.py` — `SSD512MinimalHandPose`: detect hands with SSD512, then
  estimate pose per hand. Works on an `--image` or the webcam.
- `demo3D.py` — `SSD512MinimalHandPose` with a live matplotlib 3D keypoint plot.
- `is_open_demo.py` — `ClassifyHandClosure`: label the hand OPEN or CLOSE.

## Examples

```bash
python demo_image.py
python detect_demo.py --image path/to/image.png
python demo.py --camera_id 0
python demo3D.py --camera_id 0
```

Pass `--right_hand` for right-hand inference.
