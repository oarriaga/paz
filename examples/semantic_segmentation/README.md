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

## Cityscapes

`train_cityscapes.py` / `demo_cityscapes.py` train the same UNet on real
**Cityscapes** street scenes at the **8-category** level (`void`, `flat`,
`construction`, `object`, `nature`, `sky`, `human`, `vehicle`) — exactly the
official Cityscapes `categoryId` grouping. `void` is class 0 (a real channel),
so the Dice loss applies directly with no ignore-index handling.

Cityscapes is login-gated. Two ways to fetch it:

- **Official:** `paz.datasets.cityscapes.download(root)` uses the maintained
  downloader — `pip install cityscapesScripts` and export `CITYSCAPES_USERNAME` /
  `CITYSCAPES_PASSWORD` from a free account at https://www.cityscapes-dataset.com.
- **Kaggle mirror:** `paz.datasets.cityscapes.download_kaggle(root)` pulls a
  mirror that carries the official `_gtFine_labelIds.png` and returns the dataset
  root (needs a Kaggle API token; it symlinks `leftImg8bit` to the mirror's
  `images` folder so the loader sees the official layout).

Either way the layout the loader reads is:

```
<root>/leftImg8bit/{train,val,test}/<city>/<frame>_leftImg8bit.png
<root>/gtFine/{train,val,test}/<city>/<frame>_gtFine_labelIds.png
```

```bash
# --download fetches the data first (needs the env credentials above)
python train_cityscapes.py --root /data/cityscapes --image_size 256 --download
python demo_cityscapes.py --image scene_leftImg8bit.png \
    --weights experiments/unet_cityscapes.weights.h5
```

`paz.datasets.cityscapes.load(root, split)` returns paired image/label paths
(lazy); the training `PyDataset` loads, resizes (bilinear images,
nearest-neighbor labels), and remaps raw label IDs to the 8 categories per batch.

## Notes

- The model emits `(H, W, num_classes)`; `argmax` over the channel axis gives the
  class map, drawn with `paz.draw.overlay_masks`.
- VGG preprocessing (RGB→BGR, ImageNet mean subtraction) matches the backbone.
- No trained weights are hosted for either dataset — run the train scripts to
  produce them. A `paz.applications` inference entry can be wired once a
  checkpoint is uploaded.
