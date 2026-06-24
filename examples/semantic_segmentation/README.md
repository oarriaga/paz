# Semantic segmentation (UNet)

Trains a UNet (VGG16 backbone) for per-pixel semantic segmentation on the
synthetic **Shapes** dataset that ships with `paz` — no download required. Each
pixel is classified as `background`, `square`, `circle`, or `triangle`
(`num_classes = 4`, softmax output). The masks are visualised by blending
per-class colors over the image.

## Train

```bash
python train.py --save_path experiments
```

Generates Shapes samples, builds `paz.models.UNET_VGG16` (ImageNet backbone),
trains with the Dice loss (`paz.losses.dice`; `jaccard` and `focal` are also
available), and saves `experiments/unet_shapes.weights.h5`.

## Demo

```bash
python demo.py --weights experiments/unet_shapes.weights.h5   # or --image face.jpg
```

With no `--image`, a fresh Shapes sample is generated, segmented, and the
colored masks are blended onto it.

## Notes

- The model emits `(H, W, num_classes)`; `argmax` over the channel axis gives the
  class map, drawn with `paz.draw.overlay_masks`.
- VGG preprocessing (RGB→BGR, ImageNet mean subtraction) matches the backbone.
- No trained weights are hosted — run `train.py` to produce them. A
  `paz.applications` inference entry can be wired once a checkpoint is uploaded.
- Cityscapes (the legacy example's other path) is omitted: it needs a dataset
  download and a new loader, so it is not verifiable here.
