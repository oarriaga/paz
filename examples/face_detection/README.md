# Face detection

Runs two Haar cascade detectors on a webcam stream and draws the boxes
of each detector in its own color.

## Run

```bash
python3 demo.py
python3 demo.py --models frontalface_default eye
```

Options: `--camera` (default 0), `--H` (480), `--W` (640), `--models`
(two cascade names, default `frontalface_default eye`). The script also
parses `--image_path` but does not use it.

## Output

A video window with the webcam stream and the drawn boxes.

## Requirements

A webcam; the script cannot run on an image file.
