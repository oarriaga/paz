import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

from model import custom_mse

model = tf.keras.models.load_model('/media/fabian/Data/Masterarbeit/dope_model_epoch_75.pkl', custom_objects={'custom_mse': custom_mse })
print(model.summary())

for layer in model.layers:
    if type(layer) == tf.keras.layers.Conv2D:
        print(layer.activation)