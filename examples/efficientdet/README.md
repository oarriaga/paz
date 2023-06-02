### This example shows the following:
#### &emsp;  1. How to use EfficientDet models out of the box for object detection. (explained in [section-1](#1-using-coco-pre-trained-efficientdet-models))
#### &emsp;  2. How to train or fine-tune an EfficientDet model on a custom dataset. (explained in [section-2](#2-training-or-fine-tuning-efficientdet-models-on-custom-dataset))
---

### 1. Using COCO pre-trained EfficientDet models
* `paz` contains Efficientdet models EfficientDetD0, EfficientDetD1, EfficientDetD2, ... until EfficientDetD7 that are pre-trained on COCO dataset and are ready to be used.
* An example usage of COCO pre-trained EfficientDetD0 model is shown in `demo.py` python script.
* To run the inference simply run the following command:
```
python demo.py
```
* To test the live object detection from camera, run:
```
python demo_video.py
```

* To perform inference using larger EfficientDet models, replace `EFFICIENTDETD0COCO` with `EFFICIENTDETDXCOCO` in the `demo.py` or `demo_video.py` script, where X in `EFFICIENTDETDXCOCO` can take values from (0,) 1 to 7.
* In this way any of the readily available COCO pre-trained EfficientDet model can be used for inference.

---

### 2. Training (or fine tuning) EfficientDet models on custom dataset
* To train or fine tune an EfficientDet model on a custom dataset, you may wish to use COCO pretrained weights rather than training the model from scratch.
* To do so, in the `train.py` script set the `base_weights` to `'COCO'` and `head_weights` to `None`.
* Replace `num_classes` by a value that indicates the number of classes in the custom dataset that one wants to train the model on.

The following gives an example on how to train an EfficientDetD0 model on VOC dataset.
1. In this same directory download the VOC dataset into a folder named `VOCdevkit`.
2. In the `train.py` script replace the `num_classes` by the number of classes in the VOC dataset i.e 21.
3. Further in the line where the EfficientDet model is instantiated set `base_weights` to `'COCO`' and `head_weights` to `None`.
4. Any optimizer and a suitable loss function can be chosen. By default `SGD` optimizer and `MultiBoxLoss` from `paz.optimization` is used.
5. You may also choose training parameters such as `batch_size`, `learning_rate`, `momentum` etc, according to your application. Default values are used when they not explicitly specified.
6. To start the training run the following command.
```
python train.py
```
---
