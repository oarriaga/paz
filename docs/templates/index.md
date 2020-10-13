# Introduction
PAZ is a hierarchical perception library in Python.

## Selected examples:
PAZ is used in the following examples (links to **real-time demos** and training scripts):

| [Probabilistic 2D keypoints](https://github.com/oarriaga/paz/tree/master/examples/probabilistic_keypoint_estimation)| [6D head-pose estimation](https://github.com/oarriaga/paz/tree/master/examples/pose_estimation)  | [Object detection](https://github.com/oarriaga/paz/tree/master/examples/object_detection)|
|---------------------------|--------------------------| ------------------|
|<img src="https://raw.githubusercontent.com/oarriaga/altamira-data/master/images/probabilistic_keypoints.png" width="380" height="400"> | <img src="https://raw.githubusercontent.com/oarriaga/altamira-data/master/images/head_pose.png" width="390" height="400">| <img src="https://raw.githubusercontent.com/oarriaga/altamira-data/master/images/object_detection.png" width="400" height="400">|

| [Emotion classifier](https://github.com/oarriaga/paz/tree/master/examples/face_classification) | [2D keypoint estimation](https://github.com/oarriaga/paz/tree/master/examples/keypoint_estimation)   | [Mask-RCNN (in-progress)](https://github.com/oarriaga/paz/tree/mask_rcnn/examples/mask_rcnn)  |
|---------------------------|--------------------------| -----------------------|
|<img src="https://raw.githubusercontent.com/oarriaga/altamira-data/master/images/emotion.gif" width="400" height="400">| <img src="https://raw.githubusercontent.com/oarriaga/altamira-data/master/images/keypoints.png" width="392" height="400">| <img src="https://raw.githubusercontent.com/oarriaga/altamira-data/master/images/mask.png" width="400" height="400">|

| [3D keypoint discovery](https://github.com/oarriaga/paz/tree/master/examples/discovery_of_latent_keypoints)     | [Haar Cascade detector](https://github.com/oarriaga/paz/tree/master/examples/haar_cascade_detectors) | 6D pose estimation |
|---------------------------|-----------------------| --------------------------|
|<img src="https://raw.githubusercontent.com/oarriaga/altamira-data/master/images/discovery_keypoints.png" width="400" height="400"> | <img src="https://raw.githubusercontent.com/oarriaga/altamira-data/master/images/haar_cascades.png" width="400" height="400">| <img src="https://raw.githubusercontent.com/oarriaga/altamira-data/master/images/pose_estimation.png" width="400" height="400"> |

| [Implicit orientation](https://github.com/oarriaga/paz/tree/master/examples/implicit_orientation_learning)  | [Attention (STNs)](https://github.com/oarriaga/paz/tree/master/examples/spatial_transfomer_networks) |     |
|---------------------------|-----------------------|-----------------|
|<img src="https://raw.githubusercontent.com/oarriaga/altamira-data/master/images/implicit_pose.png" width="330">| <img src="https://raw.githubusercontent.com/oarriaga/altamira-data/master/images/attention.png" width="340"> | <img src="https://raw.githubusercontent.com/oarriaga/altamira-data/master/images/your_example_here.png" width="400">|


All models can be re-trained with your own data (except for Mask-RCNN, we are working on it [here](https://github.com/oarriaga/paz/tree/mask_rcnn)).

## Hierarchical APIs
PAZ can be used with three diferent API levels which are there to be helpful for the user's specific application.

## High-level
Easy out-of-the-box prediction. For example, for detecting objects we can call the following pipeline:

``` python
from paz.pipelines import SSD512COCO

detect = SSD512COCO()

# apply directly to an image (numpy-array)
inferences = detect(image)
```

There are multiple high-level functions a.k.a. ``pipelines`` already implemented in PAZ [here](https://github.com/oarriaga/paz/tree/master/paz/pipelines). Those functions are build using our mid-level API described now below.

<p align="center">
<img src="https://raw.githubusercontent.com/oarriaga/altamira-data/master/images/object_detections_in_the_street.png" width="1000">
</p>


## Mid-level
While the high-level API is useful for quick applications, it might not be flexible enough for your specific purporse. Therefore, in PAZ we can build high-level functions using our a mid-level API.

### Mid-level: Sequential
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

<p align="center">
   <img src="https://raw.githubusercontent.com/oarriaga/altamira-data/master/images/examples_of_image_augmentation.png" width="800">
</p>


You can also add **any function** not only those found in ``processors``. For example we can pass a numpy function to our original data-augmentation pipeline:

``` python
augment.add(np.mean)
```

There are multiple functions a.k.a. ``Processors`` already implemented in PAZ [here](https://github.com/oarriaga/paz/tree/master/paz/processors).

Using these processors we can build more complex pipelines e.g. **data augmentation for object detection**: [``pr.AugmentDetection``](https://github.com/oarriaga/paz/blob/master/paz/pipelines/detection.py#L46)

<p align="center">
   <img src="https://raw.githubusercontent.com/oarriaga/altamira-data/master/images/examples_of_object_detection_augmentation.png" width="800">
</p>



### Mid-level: Explicit
Non-sequential pipelines can be also build by abstracting ``Processor``. In the example below we build a emotion classifier from scratch using our high-level and mid-level functions.

``` python
from paz.applications import HaarCascadeFrontalFace, MiniXceptionFER
import paz.processors as pr

class EmotionDetector(pr.Processor):
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
<p align="center">
   <img src="https://raw.githubusercontent.com/oarriaga/altamira-data/master/images/emotion_classification_in_the_wild.png" width="800">
</p>


``Processors`` allow us to easily compose, compress and extract away parameters of functions. However, most processors are build using our low-level API (backend) shown next.


## Low-level

Mid-level processors are mostly built from small backend functions found in: [boxes](https://github.com/oarriaga/paz/blob/master/paz/backend/boxes.py), [cameras](https://github.com/oarriaga/paz/blob/master/paz/backend/camera.py), [images](https://github.com/oarriaga/paz/tree/master/paz/backend/image), [keypoints](https://github.com/oarriaga/paz/blob/master/paz/backend/keypoints.py) and [quaternions](https://github.com/oarriaga/paz/blob/master/paz/backend/quaternion.py).

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

## Additional functionality

* PAZ has [built-in messages](https://github.com/oarriaga/paz/blob/master/paz/abstract/messages.py) e.g. ``Pose6D`` for an easier data exchange with other libraries or frameworks such as [ROS](https://www.ros.org/).

* There are custom [callbacks](https://github.com/oarriaga/paz/blob/master/paz/optimization/callbacks.py) e.g. MAP evaluation for object detectors while training
    
* PAZ comes with [data loaders](https://github.com/oarriaga/paz/tree/master/paz/datasets) for the multiple datasets:
    [OpenImages](https://github.com/oarriaga/paz/blob/master/paz/datasets/open_images.py), [VOC](https://github.com/oarriaga/paz/blob/master/paz/datasets/voc.py), [YCB-Video](https://github.com/oarriaga/paz/blob/master/examples/object_detection/datasets/ycb_video.py), [FAT](https://github.com/oarriaga/paz/blob/master/paz/datasets/fat.py), [FERPlus](https://github.com/oarriaga/paz/blob/master/paz/datasets/ferplus.py), [FER2013](https://github.com/oarriaga/paz/blob/master/paz/datasets/fer.py).

* We have an automatic [batch creation and dispatching wrappers](https://github.com/oarriaga/paz/blob/master/paz/abstract/sequence.py) for an easy connection between you ``pipelines`` and tensorflow generators. Please look at the examples and the processor ``pr.SequenceWrapper`` for more information.
