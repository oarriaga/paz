import numpy as np
from tensorflow.python.data import Dataset
import tensorflow as tf
keras = tf.keras
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
import matplotlib.pyplot as plt

from paz.datasets import VVAD_LRS3
from paz.models.classification import CNN2Plus1D

# sample = FeatureizedSample()
# sample.featureType = 'faceImage'
# sample.shape = (38, 96, 96, 3)
# sample.k = 38  # Number of frames per sample

data = VVAD_LRS3().load_data()
print(data.keys())
x_test = data.get("x_test")
y_test = data.get("y_test")
x_train = data.get("x_train")
y_train = data.get("y_train")
# x_val = data.get("x_validation")

print(x_test.dtype)
print(x_test.shape)
print(x_train.shape[0])

# for i in range(10):
#     print(f"Sample number: {x_train.shape[0] // 2 + 1} label: {y_train[x_train.shape[0] // 2 + 1]}")
# train_ds = Dataset()

# size = x_train.shape[0]
# data_ds = Dataset.from_tensor_slices((x_train, y_train))
size = x_test.shape[0]
data_ds = Dataset.from_tensor_slices((x_test, y_test))

# split datasets into train and val with equally sized positive and negative datasets
neg_data_ds = data_ds.take(size // 2).shuffle(buffer_size=size // 2)
pos_data_ds = data_ds.skip(size // 2).shuffle(buffer_size=size // 2)
val_ds = neg_data_ds.take(size // 10).concatenate(pos_data_ds.take(size // 10))
train_ds = neg_data_ds.skip(size // 10).concatenate(pos_data_ds.skip(size // 10))

val_ds = val_ds.batch(8)
train_ds = train_ds.batch(8)

model = CNN2Plus1D()

loss = BinaryCrossentropy(from_logits=True)  # Alternative vor two label Classifications: Hinge Loss or Squared Hinge Loss
optimizer = Adam(learning_rate=0.0001)

model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

history = model.fit(x = train_ds,
                    epochs = 2,
                    validation_data = val_ds)


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