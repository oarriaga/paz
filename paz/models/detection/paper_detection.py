'''
This module provides the paper detection models
'''

# System imports
from pathlib import Path

# 3rd party imports
from tensorflow.keras.models import load_model


# local imports

# end file header
__author__ = 'Adrian Lubitz'

DETECTION_MODEL_PATH = Path('/home/alubitz', 'repos', 'paperdetector',
                            'models', 'corner_detection.v0.0.0')  # TODO: download
REFINER_MODEL_PATH = Path('/home/alubitz', 'repos', 'paperdetector',
                          'models', 'corner_refiner.v0.0.0')  # TODO: download


def PaperDetection():
    model = load_model(DETECTION_MODEL_PATH)
    return model


def PaperRefiner():
    model = load_model(REFINER_MODEL_PATH)
    return model
