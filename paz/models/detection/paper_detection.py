'''
This module provides the paper detection models
'''

# System imports
from pathlib import Path
import zipfile

# 3rd party imports
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import get_file


# local imports

# end file header
__author__ = 'Adrian Lubitz'

CACHE_SUBDIR = Path('paz', 'models')

DETECTION_MODEL_URL = 'https://github.com/adrianlubitz/open_source_models/releases/download/v0.0.0/corner_detection.zip'
DETECTION_MODEL_PATH = Path(CACHE_SUBDIR, 'corner_detection')
REFINER_MODEL_URL = 'https://github.com/adrianlubitz/open_source_models/releases/download/v0.0.0/corner_refiner.zip'
REFINER_MODEL_PATH = Path(CACHE_SUBDIR, 'corner_refiner')


def PaperDetection():
    path = get_file(origin=DETECTION_MODEL_URL, extract=True, cache_subdir=CACHE_SUBDIR)
    model = load_model(Path(path).with_suffix(''))
    return model


def CornerRefiner():
    path = get_file(origin=REFINER_MODEL_URL, extract=True, cache_subdir=CACHE_SUBDIR)
    model = load_model(Path(path).with_suffix(''))
    return model

def build_model(model_name, **kwargs):
    """
    loads the model with the corresponding model name
    Args:
        model_name: Name of the model. Can be of 'corner_detection', 'corner_refiner'
    Returns:
        model: a keras model
    """
    if model_name.lower() == 'corner_detection':
        return build_get_corners(**kwargs)
    if model_name.lower() == 'corner_refiner':
        return build_refine_corner(**kwargs)


def build_get_corners():  # TODO:support arguments for shape and maybe others
    '''
    rebuild the corner detection model from https://khurramjaved.com/RecursiveCNN.pdf 
    '''
    # TODO: more modern approach is maybe to use seperabelConv2D: https://keras.io/examples/vision/keypoint_detection/
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(32, 32, 3)))
    # Conv 1
    model.add(layers.Conv2D(kernel_size=(5, 5), activation='relu', filters=20,
              padding='same'))  # TODO: maybe use 3D filter instead of 2D?
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    # Conv 2
    model.add(layers.Conv2D(kernel_size=(5, 5),
              activation='relu', filters=40, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(kernel_size=(5, 5),
              activation='relu', filters=40, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    # Conv 3
    model.add(layers.Conv2D(kernel_size=(5, 5),
              activation='relu', filters=60, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(kernel_size=(5, 5),
              activation='relu', filters=60, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    # Conv 4
    model.add(layers.Conv2D(kernel_size=(5, 5),
              activation='relu', filters=80, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    # Conv 5
    model.add(layers.Conv2D(kernel_size=(5, 5),
              activation='relu', filters=100, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    # Conv 6
    model.add(layers.Conv2D(kernel_size=(5, 5),
              activation='relu', filters=100, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    # Fully Connected 1
    # TODO: Paper stated 0.8 dropout rate
    model.add(layers.Dense(500, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(500, activation='relu'))
    model.add(layers.Dropout(0.5))
    # Sigmoid Activation
    # regression is needed here -> sigmoid
    model.add(layers.Dense(8, activation='sigmoid'))
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam', metrics=[
                  'mse', 'mae'])  # metrics from https://keras.io/api/metrics/regression_metrics/
    model._name = 'corner_detection'
    return model


def build_refine_corner():  # TODO:support arguments for shape and maybe others
    '''
    rebuild the corner refiner model from https://khurramjaved.com/RecursiveCNN.pdf 
    '''
    # TODO: more modern approach is maybe to use seperabelConv2D: https://keras.io/examples/vision/keypoint_detection/
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(32, 32, 3)))
    # Conv 1
    model.add(layers.Conv2D(kernel_size=(5, 5), activation='relu', filters=10,
              padding='same'))  # TODO: maybe use 3D filter instead of 2D?
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    # Conv 2
    model.add(layers.Conv2D(kernel_size=(5, 5),
              activation='relu', filters=10, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    # Conv 3
    model.add(layers.Conv2D(kernel_size=(5, 5),
              activation='relu', filters=20, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    # Conv 4
    model.add(layers.Conv2D(kernel_size=(5, 5),
              activation='relu', filters=30, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    # Conv 5
    model.add(layers.Conv2D(kernel_size=(5, 5),
              activation='relu', filters=40, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    # Fully Connected 1
    # TODO: Paper stated 0.8 dropout rate
    model.add(layers.Dense(300, activation='relu'))
    model.add(layers.Dropout(0.5))
    # Sigmoid Activation
    # regression is needed here -> sigmoid
    model.add(layers.Dense(2, activation='sigmoid'))
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam', metrics=[
                  'mse', 'mae'])  # metrics from https://keras.io/api/metrics/regression_metrics/
    model._name = 'corner_refiner'
    return model