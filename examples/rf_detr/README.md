# RF-DETR

`detect.py` runs COCO object detection with RF-DETR. `fine_tune.py`
fine-tunes a COCO RF-DETR on VOC2007; `generator.py` builds its batches.

## Run

```bash
python3 detect.py
python3 fine_tune.py --freeze_backbone
```

`detect.py` options: `--variant` (default `nano`), `--score_thresh`
(0.5), `--image`, `--output` (default `detections.png`). With no
arguments it downloads the pretrained nano weights and a demo image.

`fine_tune.py` options: `--variant`, `--batch_size` (4),
`--learning_rate` (1e-4), `--epochs` (30), `--max_num_boxes` (25),
`--freeze_backbone`, `--steps_per_epoch`, `--validation_steps`, `--root`
(default `experiments`), `--label`.

## Output

`detect.py` writes the image with boxes drawn to `--output`.
`fine_tune.py` writes `rf_detr_<variant>_voc.weights.h5` and
`optimization.log` in the experiment directory under `--root`.

## Requirements

`fine_tune.py` needs the VOC2007 dataset.
