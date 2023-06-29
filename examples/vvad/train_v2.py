import numpy as np
from tensorflow.python.data import Dataset
import tensorflow as tf
keras = tf.keras
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
import matplotlib.pyplot as plt

from paz.datasets import VVAD_LRS3
from paz.models.classification import CNN2Plus1D

generatorTrain = VVAD_LRS3(split="train")
generatorVal = VVAD_LRS3(split="val")

datasetTrain = Dataset.from_generator(generatorTrain, output_signature=(tf.TensorSpec(shape=(38, 96, 96, 3)), tf.TensorSpec(shape=(), dtype=tf.int8)))
datasetVal = Dataset.from_generator(generatorVal, output_signature=(tf.TensorSpec(shape=(38, 96, 96, 3)), tf.TensorSpec(shape=(), dtype=tf.int8)))

datasetTrain = datasetTrain.batch(8)
datasetVal = datasetVal.batch(8)

model = CNN2Plus1D()

loss = BinaryCrossentropy(from_logits=True)  # Alternative for two label Classifications: Hinge Loss or Squared Hinge Loss
optimizer = Adam(learning_rate=0.0001)

model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

history = model.fit(x = datasetTrain,
                    epochs = 2,
                    validation_data = datasetVal)


def plot_history(history):
  """
    Plotting training and validation learning curves.

    Args:
      history: model history with all the metric measures
  """
  fig, (ax1, ax2) = plt.subplots(2)

  fig.set_size_inches(18.5, 10.5)

  # Plot loss
  ax1.set_title('Loss')
  ax1.plot(history.history['loss'], label = 'train')
  ax1.plot(history.history['val_loss'], label = 'test')
  ax1.set_ylabel('Loss')

  # Determine upper bound of y-axis
  max_loss = max(history.history['loss'] + history.history['val_loss'])

  ax1.set_ylim([0, np.ceil(max_loss)])
  ax1.set_xlabel('Epoch')
  ax1.legend(['Train', 'Validation'])

  # Plot accuracy
  ax2.set_title('Accuracy')
  ax2.plot(history.history['accuracy'],  label = 'train')
  ax2.plot(history.history['val_accuracy'], label = 'test')
  ax2.set_ylabel('Accuracy')
  ax2.set_ylim([0, 1])
  ax2.set_xlabel('Epoch')
  ax2.legend(['Train', 'Validation'])

  plt.show()

plot_history(history)