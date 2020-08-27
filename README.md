# (PAZ) Perception for Autonomous Systems
![Python package](https://github.com/oarriaga/paz/workflows/Python%20package/badge.svg)

Hierarchical perception library in Python.

## Selected examples:
PAZ is used in the following examples:

| [Probabilistic 2D keypoints](https://github.com/oarriaga/paz/tree/master/examples/probabilistic_keypoint_estimation)| [6D head-pose estimation](https://github.com/oarriaga/paz/tree/master/examples/pose_estimation)  | [Object detection](https://github.com/oarriaga/paz/tree/master/examples/object_detection)|
|---------------------------|--------------------------| ------------------|
|<img src="https://github.com/oarriaga/altamira-data/blob/master/images/probabilistic_keypoints.png" width="400"> | <img src="https://github.com/oarriaga/altamira-data/blob/master/images/head_pose.png" width="400">| <img src="https://github.com/oarriaga/altamira-data/blob/master/images/object_detection.png" width="430">|

| [Emotion classifier](https://github.com/oarriaga/paz/tree/master/examples/face_classification) | [2D keypoint estimation](https://github.com/oarriaga/paz/tree/master/examples/keypoint_estimation)   | [Mask-RCNN (in-progress)](https://github.com/oarriaga/paz/tree/master/examples/mask_rcnn)  |
|---------------------------|--------------------------| -----------------------|
|<img src="https://github.com/oarriaga/altamira-data/blob/master/images/emotion.gif" width="400">| <img src="https://github.com/oarriaga/altamira-data/blob/master/images/keypoints.png" width="420">| <img src="https://github.com/oarriaga/altamira-data/blob/master/images/mask.png" width="400">|

| [3D keypoint discovery](https://github.com/oarriaga/paz/tree/master/examples/discovery_of_latent_keypoints)     | [Haar Cascade detector](https://github.com/oarriaga/paz/tree/master/examples/haar_cascade_detectors) | 6D pose estimation |
|---------------------------|-----------------------| --------------------------|
|<img src="https://github.com/oarriaga/altamira-data/blob/master/images/discovery_keypoints.png" width="410"> | <img src="https://github.com/oarriaga/altamira-data/blob/master/images/haar_cascades.png" width="410">| <img src="https://github.com/oarriaga/altamira-data/blob/master/images/pose_estimation.png" width="400"> |

| [Implicit orientation](https://github.com/oarriaga/paz/tree/master/examples/implicit_orientation_learning)  | [Attention (STNs)](https://github.com/oarriaga/paz/tree/master/examples/spatial_transfomer_networks) |
|---------------------------|-----------------------|
|<img src="https://github.com/oarriaga/altamira-data/blob/master/images/implicit_pose.png" width="512">|<img src="https://github.com/oarriaga/altamira-data/blob/master/images/attention.png" width="512"> |

All models can be re-trained with your own data (except for Mask-RCNN, we are working on it [here](https://github.com/oarriaga/paz/tree/mask_rcnn)).

## Table of Contents
<!--ts-->
* [Examples](#selected-examples)
* [Hierarchical APIs](#hierarchical-apis)
    * [High-level](#high-level) | [Mid-level](#mid-level) | [Low-level](#mid-level)
* [Additional functionality](#additional-functionality)
* [Installation](#installation)
* [Motivation](#motivation)
<!--te-->

## Hierarchical APIs
PAZ can be used with three diferent API levels which are there to be helpful for the user's specific application.

### High-level
Easy out-of-the-box prediction. For example, for detecting objects we can call the following pipeline:

``` python
from paz.pipelines import SSD512COCO

detect = SSD512COCO()

# apply directly to an image (numpy-array)
inferences = detect(image)
```

There are multiple high-level functions a.k.a. ``pipelines`` already implemented in PAZ [here](https://github.com/oarriaga/paz/tree/master/paz/pipelines). Those functions are build using our mid-level API described now below.

### Mid-level
While the high-level API is useful for quick applications, it might not be flexible enough for your specific purporse. Therefore, in PAZ we can build high-level functions using our a mid-level API.

#### Mid-level: Sequential
If your function is sequential you can construct a sequential function using ``SequentialProcessor``. In the example below we create a data-augmentation pipeline:

``` python
from paz.abstract import SequentialProcessor
from paz import processors as pr

augment = SequentialProcessor()
augment.add(pr.RandomContrast())
augment.add(pr.RandomBrightness())
augment.add(pr.RandomSaturation())
augment.add(pr.RandomHue())

# you can now use this now as a normal function
image = augment(image)
```

You can also add **any function** not only those found in ``processors``. For example we can pass a numpy function to our original data-augmentation pipeline:

``` python
augment.add(np.mean)
```
There are multiple functions a.k.a. ``Processors`` already implemented in PAZ [here](https://github.com/oarriaga/paz/tree/master/paz/processors)


#### Mid-level: Explicit
Non-sequential pipelines can be also build by abstracting ``Processor``. In the example below we build a emotion classifier from scratch using our high-level and mid-level functions.

``` python
from paz.pipelines import HaarCascadeFrontalFace, MiniXceptionFER
from paz.abstract import Processor
import paz.processors as pr

class EmotionDetector(Processor):
    def __init__(self):
        super(EmotionDetector, self).__init__()
        self.detect = HaarCascadeFrontalFace(draw=False)
        self.crop = pr.CropBoxes2D()
        self.classify = MiniXceptionFER()
        self.draw = pr.DrawBoxes2D(self.classify.class_names)

    def call(self, image):
        boxes2D = self.detect(image)['boxes2D']
        cropped_images = self.crop(image, boxes2D)
        for cropped_image, box2D in zip(cropped_images, boxes2D):
            box2D.class_name = self.classify(cropped_image)['class_name']
        return self.draw(image, boxes2D)
        
detect = EmotionDetector()
# you can now apply it to an image (numpy array)
predictions = detect(image)
```

``Processors`` allow us to easily compose, compress and extract away unecessary parameters of functions. However, processors are ultimately build using our low-level API (small functions) explained next.

### Low-level

PAZ has a lot of small functions for [boxes](https://github.com/oarriaga/paz/blob/master/paz/backend/boxes.py), [cameras](https://github.com/oarriaga/paz/blob/master/paz/backend/camera.py), [images](https://github.com/oarriaga/paz/tree/master/paz/backend/image), [keypoints](https://github.com/oarriaga/paz/blob/master/paz/backend/keypoints.py) and [quaternions](https://github.com/oarriaga/paz/blob/master/paz/backend/quaternion.py).

These functions can found in ``paz.backend``:

``` python
from paz.backend import boxes, camera, image, keypoints, quaternion
```
For example, you can use them in your scripts to load or show images:

``` python
from paz.backend.image import load_image, show_image

image = load_image('my_image.png')
show_image(image)
```

### Additional functionality

* PAZ has [built-in messages](https://github.com/oarriaga/paz/blob/master/paz/abstract/messages.py) e.g. ''Pose6D'' for easier data exchange with other libraries or frameworks such as [ROS](https://www.ros.org/).

* There are custom callbacks e.g. MAP evaluation for object detectors while training
    
* PAZ comes with data loaders for the following datasets:
    OpenImages, VOC, YCB-Video, FAT, FERPlus, FER2013

* We have an automatic batch creation and dispatching wrappers for an easy connection between generators and our pipelines. Please look at the examples and the processor ``pr.SequenceWrapper`` for more information.

The following models are implemented in PAZ. All models can be trained with your own data.

<center>

| Task (link to tutorial)    |Model (link to paper)  |
|---------------------------:|-----------------------| 
|Object detection            |SSD-512                |
|Object detection            |SSD-300                |
|Probabilistic keypoint est. |Gaussian Mixture CNN   |
|Detection and Segmentation  |MaskRCNN (in progress) |
|Keypoint estimation         |HRNet                  |
|6D Pose estimation          |KeypointNet2D          |
|Implicit orientation        |AutoEncoder            |
|Emotion classification       |MiniXception           |
|Discovery of Keypoints      |KeypointNet            |
|Keypoint estimation         |KeypointNet2D          |
|Attention                   |Spatial Transformers   |
|Object detection            |HaarCascades           |

</center>

## Installation
PAZ has only **three** dependencies: [Tensorflow2.0](https://www.tensorflow.org/), [OpenCV](https://opencv.org/) and [NumPy](https://numpy.org/).

To install PAZ you can run:

```
pip install . --user
```

## Coverage
Test coverage can be checked using [coverage](https://coverage.readthedocs.io/en/coverage-5.2.1/).
You can install coverage by calling: `pip install coverage --user`
You can then check for the test coverage by running:
```
coverage run -m pytest tests/
coverage report -m
```

## Motivation
Even though there are multiple high-level computer vision libraries in different DL frameworks, I felt there was not a consolidated deep learning library for robot-perception in my framework of choice (Keras).

### Why Keras over other frameworks/libraries?
In simple terms, I have always felt the API to be more mature.
It allowed me to express my ideas at the level of complexity that was required. 
Keras was often misinterpreted as a "beginners" framework; however, once you learn to abstract: Layer, Callbacks, Loss, Metrics or Model, the API remained intact and helpful for more complicated ideas. 
It allowed me to automate and write down experiments with no extra boilerplate code.
Furthermore, if someone wanted to abandon such comfort one could still create a custom training loop.

As a final remark, I would like to mention, that I feel that we might tend to forget the great effort and emotional carriage behind every open-source project.
I feel it's easy to blurry a company name with the individuals behind their project, and we forget that there is someone feeling our criticism and our praise.
Therefore, whatever good code you can find here, is all dedicated to the software-engineers and contributors of open-source projects like Pytorch, Tensorflow and Keras.
You put your craft out there for all of us to use and appreciate, and we ought first to give you our thankful consideration before we lay upon you our hardened criticism.

## Why PAZ?
