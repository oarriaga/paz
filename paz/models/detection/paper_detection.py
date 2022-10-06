'''
This module provides the paper detection models
'''

# System imports
from pathlib import Path

# 3rd party imports
from tensorflow.keras.models import load_model


# local imports

# end file header
__author__      = 'Adrian Lubitz'

DETECTION_MODEL_PATH = Path('~', 'repos', 'paperdetector', 'models', 'corner_detection.v0.0.dev20221006-074222') #TODO: download

def PaperDetection():
    model = load_model(DETECTION_MODEL_PATH)
    return model