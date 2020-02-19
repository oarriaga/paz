# (PAZ) Perception for Autonomous Systems

PAZ is a high-level robot perception python library.

## Installation
PAZ has two dependencies: [Tensorflow2.0](https://www.tensorflow.org/) and [OpenCV](https://opencv.org/).

### GPU installation
1. Please consult the [tensorflow documentation](https://www.tensorflow.org/install/gpu) and install their [requirements](https://www.tensorflow.org/install/gpu#software_requirements) (NVIDIA drivers, CUDA 10.1, cuDNN)

2. Change the dependency ``tensorflow`` in setup.py to ``tensorflow-gpu``

3. Run: `pip install . --user`

### CPU installation
1. Run: `pip install . --user`

### Common issues
OpenCV is automatically installed using [opencv-python](https://github.com/skvark/opencv-python) wheels.
In case the pre-compiled versions are not working for you, try to compile OpenCV from scratch.
