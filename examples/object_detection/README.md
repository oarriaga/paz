# Object detection

Training, fine-tuning and evaluation of detectors on VOC, plus helper
modules (`generator.py`, `pipeline.py`, `compute_MAP.py`) and scratch
scripts.

## Run

```bash
python3 train.py --model SSD300
python3 evaluate.py --model SSD300VOC
python3 augment_demo.py
```

- `train.py`: trains on VOC2007 and VOC2012 `trainval` and tests on
  VOC2007 `test`. It defaults to the Keras torch backend. Options
  include `--model` (default `SSD300`), `--root` (`experiments`),
  `--label`, `--batch_size` (32), `--learning_rate` (0.001),
  `--num_iterations` (120000), `--H`, `--W`, `--eval_period` (10); see
  `--help` for the rest. Writes `<model>.keras` and `optimization.log`
  in the experiment directory under `--root`.
- `evaluate.py`: `--model` (`SSD300VOC` or `EFFICIENTDETD0VOC`),
  `--dataset` (default `VOC2007`), `--split` (`test`), `--score_thresh`,
  `--nms_thresh`, `--iou_thresh`, `--top_k`, `--use_07_metric`. Prints
  the average precision per class and the mAP.
- `augment_demo.py`: no options; loads VOC2007 `trainval` and writes
  `augmentations.jpg`.
- `fine_tune.py`: SSD fine-tuning; run `--help` for its options.

## Requirements

The VOC2007 dataset, and VOC2012 for `train.py`.
