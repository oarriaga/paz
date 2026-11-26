# Emotion classifier

`demo.py` runs `paz.applications.DetectMiniXceptionFER` on a webcam
stream. `train.py` trains a `MiniXception` model with 48x48 grayscale
input on the `train` and `validation` splits of its FER loader.

## Run

```bash
python3 demo.py
python3 train.py --epochs 100
```

`demo.py` options: `--camera` (default 0), `--H` (480), `--W` (640). It
also parses `--image_path` but does not use it.

`train.py` options: `--root` (default `experiments`), `--batch_size`
(32), `--epochs` (100).

## Output

`demo.py` shows a video window. `train.py` writes `log.csv` and the best
model `fer.keras` in `--root`.

## Requirements

`demo.py` needs a webcam and cannot run on an image file. `train.py`
needs the dataset read by its loader.
