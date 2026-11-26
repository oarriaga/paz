# Hand detection

Hand detection with SSD512 and hand pose estimation, plus a training
script on the `Human hand` subset of Open Images V6.

## Run

```bash
python3 demo.py --image photo.jpg
python3 pose_demo.py --image photo.jpg
python3 train.py --data_path open-images-v6/
```

- `demo.py` (`paz.applications.SSD512HandDetection`): `--image` (RGB
  image; webcam if unset), `--camera` (0), `--score_thresh` (0.4), `--H`
  (480), `--W` (640), `--output` (default `hand_detection.jpg`).
- `pose_demo.py` (`paz.applications.DetectMinimalHand`): `--image`,
  `--camera`, `--box_scale` (1.5), `--right_hand`, `--H`, `--W`,
  `--output` (default `hand_pose.jpg`).
- `train.py`: `--data_path`, `--root` (default `experiments`), `--label`,
  `--batch_size` (16), `--learning_rate` (1e-4), `--num_epochs` (240),
  `--input_size` (512), `--match_IOU`, `--max_num_boxes`,
  `--num_workers`, `--max_queue_size`, `--box_variances`.

## Output

With `--image` the demos write the drawn image to `--output`; without it
they show a webcam video window. `train.py` writes `log.csv` and
`SSD512Hand.keras` in the experiment directory under `--root`.

## Requirements

A webcam when `--image` is unset. `train.py` needs Open Images V6.
