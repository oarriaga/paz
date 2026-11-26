# EfficientDet

Runs the `paz.applications.EFFICIENTDETD0COCO` detector on an image or on
a webcam stream.

## Run

```bash
python3 demo.py --image photo.jpg
python3 demo.py
```

Options: `--image` (RGB image; webcam if unset), `--camera` (default 0),
`--score_thresh` (default 0.6), `--H` (480), `--W` (640), `--output`
(default `efficientdet.jpg`).

## Output

With `--image` the image with detections is written to `--output`.
Without it a video window shows the webcam stream; this needs a webcam.
