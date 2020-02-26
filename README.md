# (PAZ) Perception for Autonomous Systems
High-level perception python library.

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
