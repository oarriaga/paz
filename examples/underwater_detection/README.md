# Underwater fish detection, stage one

`train.py` trains RF-DETR with one class, `fish`, on DeepFish, or on
DeepFish and OzFish. The three stage-one experiments of the reference
recipe are flag sets of the same script.

| Run | Flags | Data | Best checkpoint | Early stopping |
|---|---|---|---|---|
| E10 | `--variant nano --dataset deepfish` | DeepFish pooled 80/20, seed 42 | `best.weights.h5`, lowest `val_first_group_loss` | patience 10, delta 1e-3 |
| E11 | `--variant large --dataset deepfish` | same | same | same |
| E15 | `--variant large --dataset deepfish_ozfish --ozfish_root <OzFish>` | DeepFish native folders + OzFish 80/20, seed 42 | `best_total.weights.h5` | off |

Every run also needs `--run_dir`. The defaults are the reference values:
20 epochs, batch 16, no gradient accumulation, learning rate 1e-4 on a
cosine to zero with no warmup, weight decay 1e-4, gradient clip 0.1, EMA
0.993, 13 query groups, seed 42.

DeepFish is read from `~/.keras/paz/datasets/Deepfish` and downloaded
there with `gdown` if missing, so DeepFish runs need `gdown` installed.
`--ozfish_root` is the OzFish folder that holds
`frames_labelled/` and `manifests/`.

The generators always run with one worker: with more, Keras prefetches
batches before the epoch callback runs, and an epoch would silently train
on the previous epoch's order.

## Run directory and resume

A run lives in one fixed `--run_dir`. To resume after a crash or a time
limit, run the same command again. Training continues from the last
finished epoch with the same batch order. The best checkpoint threshold
and the early-stopping count are read back from `log.csv`.

- The first start writes `parameters.json`. A later start with different
  flags, step count or `max_boxes` stops with an error, because a changed
  schedule length would silently change the learning rate. `--run_dir`
  itself is not pinned and `--ozfish_root` is pinned as an absolute
  path, so a relative or absolute spelling of either one resumes.
- A run whose `final_ema.weights.h5` exists is finished. Its resume state
  is gone, so `train.py` refuses to start in it again; use a new
  `--run_dir`.
- With `--steps_per_epoch` below the number of batches, Keras keeps one
  data iterator across epochs, so a resumed short run does not repeat the
  exact batches. Full epochs are unaffected.
- An E10/E11 run that already stopped early and is killed in the few
  seconds between its last backup and the `final_ema` save trains one
  extra optimizer step and logs one extra row on resubmit: Keras runs a
  batch before it checks the stop flag. The effect is negligible.
- A leftover `backup/` next to a finished `final_ema.weights.h5` is
  harmless and can be deleted.

## Outputs in `--run_dir`

| File | What |
|---|---|
| `parameters.json` | the flags except `--run_dir`, plus `steps_per_epoch` and `max_boxes` |
| `starts.log` | one line per start, with the code SHA; `-dirty` when the checkout has local changes |
| `log.csv` | per-epoch metrics, appended across resumes; E15 adds `val_regular_mAP` and `val_ema_mAP` |
| `metrics.png` | the training metrics of the current start only; the mAP columns are not drawn |
| `best.weights.h5` | E10/E11: the lowest first-group validation loss |
| `best_regular.weights.h5`, `best_ema.weights.h5` | E15: best raw and best EMA weights by validation COCO mAP@[.5:.95] |
| `best_total.weights.h5` | E15: the better of the two |
| `final_ema.weights.h5` | the EMA weights after the last epoch; its presence marks the run finished |
| `backup/` | the resume state; removed only after `final_ema.weights.h5` is written |

Plan the disk space: every `.weights.h5` holds the full optimizer state,
about 434 MB for Nano and more for Large. E10/E11 keep four such files
(best, final EMA, backup and its previous copy) and E15 six (best raw,
best EMA, best total, final EMA, backup and its previous copy).

## Run it

CPU smoke run, Nano only, a handful of steps:

```bash
KERAS_BACKEND=jax CUDA_VISIBLE_DEVICES=-1 python train.py \
    --variant nano --dataset deepfish --run_dir smoke_e10 \
    --epochs 2 --batch_size 2 --num_groups 2 \
    --steps_per_epoch 2 --validation_steps 1
```

E15 scores COCO mAP twice per epoch, on the raw and on the EMA weights,
over the full validation set, as the reference does; it ignores
`--validation_steps`. That is slow on CPU, so give an E15 smoke run a
small validation set.

## Layer decay is off

The reference multiplies each gradient by a per-layer rate (encoder 1.5,
decay 0.8 per block, 0.7 per component) after clipping and before a plain
AdamW. Adam divides every gradient by its own running size, so those
multipliers almost cancel, and the reference runs trained with practically
no layer decay. `train.py` reproduces that behaviour with
`LayerwiseAdamW({}, ...)`, where every scale is one.

## Deviations from the reference

| Reference | Here |
|---|---|
| 13 pretrained query groups | the COCO weights carry one group; groups 2-13 start fresh |
| mixed precision | float32 |
| a step is skipped when a gradient is NaN or Inf | not ported; it guards mixed precision |
| torchvision square resize | `paz.image.crop_and_resize` in the generators; `paz.image.resize_opencv` in the E15 mAP pass |
| oversampled copies written to disk | copies made in memory, each with a fixed key |
| layer decay applied to the gradients | every rate scale is one (see above) |
| no weight decay on backbone gamma, position tables, biases, norms and embeddings | no weight decay on any bias, gamma or beta; the difference is negligible |
| K+1 class logits with a background slot | K logits, sigmoid focal loss |
| E15 selects by pycocotools AP (101 recall points, 100 detections) | `paz.evaluation.compute_COCO_mAP`: all recall points, 300 detections, integer pixel boxes (`RFDETR` returns `paz.detection.to_boxes2D`, which casts to int32, `paz/backend/detection.py:264`) |
| splits under 5 batches resampled with replacement | not ported; it only affects tiny smoke runs |
| E15 saves best raw / best EMA only when the score beats the best over both | each file is saved on its own improvement; `best_total` is the same except on an exact tie |
| validation loss of the first query group | logged as `val_first_group_loss`; Keras `val_loss` scores every group |
| different GPUs and container | timing is not comparable |
