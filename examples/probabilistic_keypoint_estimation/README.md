# Probabilistic 2D facial keypoints

Estimates 2D facial keypoints together with their spatial uncertainty. Each
keypoint is modelled as a Gaussian mixture over the image plane: the network
predicts, per keypoint, a small grid of mixture components (categorical
weight, scale, and `(x, y)` mean offset). The keypoint location is the mixture
mean and the mixture density visualises the uncertainty.

## JAX port note

The legacy example used `tensorflow_probability` (`tfpl.DistributionLambda`
inside the model and `log_prob` as the loss). This JAX/Keras-3 port keeps the
network purely convolutional: it emits the raw mixture maps and the mixture
math (mean, density, negative log-likelihood) lives in
`paz.backend.gaussian_mixture` with the training loss in
`paz.losses.gaussian_mixture_nll`. There are no pretrained weights hosted yet;
train one with `train.py` and the application loads it from the conventional
`altamira-data` URL once uploaded.

## Dataset

```bash
bash dataset_downloader.sh
```

Downloads the Kaggle Facial Keypoints Detection dataset (needs the kaggle CLI,
an API token, and accepting the competition rules).

## Train

```bash
python train.py --root dataset --save_path experiments
```

## Demo

```bash
python demo.py --image path/to/face.jpg     # or no --image for the webcam
```
