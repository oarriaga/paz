# (PAZ) Perception for Autonomous Systems
High-level perception library in python.

PAZ has only three dependencies: [Tensorflow2.0](https://www.tensorflow.org/), [OpenCV](https://opencv.org/) and [NumPy](https://numpy.org/).


## Installation

### GPU installation
1. Please consult the [tensorflow documentation](https://www.tensorflow.org/install/gpu) and install their [requirements](https://www.tensorflow.org/install/gpu#software_requirements) (NVIDIA drivers, CUDA 10.1, cuDNN)

2. Change the dependency ``tensorflow`` in setup.py to ``tensorflow-gpu``

3. Run: `pip install . --user`

### CPU installation
1. Run: `pip install . --user`

### Common issues
* OpenCV is automatically installed using [opencv-python](https://github.com/skvark/opencv-python) wheels.
In case the pre-compiled versions are not working for you, try to compile OpenCV from scratch.

### Motivation
Even though there are multiple high-level computer vision libraries in different DL frameworks, I felt there was not a consolidated deep learning library for robot-perception in my framework of choice (Keras).

#### Why Keras over other frameworks/libraries?
In simple terms, I have always felt the API to be more mature.
It allowed me to express my ideas at the level of complexity that was required. 
Keras was often misinterpreted as a "beginners" framework; however, once you learn to abstract: Layer, Callbacks, Loss, Metrics, Model the API remained intact and helpful for more complicated ideas. 
It allowed me to automate and write down experiments with no extra boiler-plate code.
Furthermore, if someone wanted to abandon such comfort one could still go the beaten path and create a custom training loop.

As final remark, I would like to mention that, I feel that we might tend to forget the great effort and emotional carriage behind every open-source project.
I feel it's easy to blurry a company name with the individuals behind their project, and we forget that there is someone feeling our criticism and our praise.
Therefore, whatever good code you can find here, is all dedicated to the software-engineers and contributors of open-source projects like Pytorch, Tensorflow and Keras.
You put your craft out there for all of us to use and appreciate, and we ought first to give you our thankful consideration before we lay our hardened criticism.

### Why the name PAZ?


