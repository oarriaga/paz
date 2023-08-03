import numpy as np
import os
import argparse
import datetime
import math
import callbacks
from pathlib import Path
# from codecarbon import OfflineEmissionsTracker
from tensorflow.python.data import Dataset
import tensorflow as tf
keras = tf.keras
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam, SGD, RMSprop
import matplotlib.pyplot as plt

from paz.datasets import VVAD_LRS3
from paz.models.classification import CNN2Plus1D, VVAD_LRS3_LSTM

parser = argparse.ArgumentParser(description='Paz VVAD Training')
parser.add_argument('-p', '--data_path', type=str,
                    default='.keras/paz/datasets',
                    help='Path from your home dir to dataset directory')
parser.add_argument('-m', '--model', type=str,
                    default='VVAD_LRS3',
                    help='Model you want to train',
                    choices=['VVAD_LRS3', 'CNN2Plus1D', 'MoViNets', 'ViViT'])
parser.add_argument('-b', '--batch_size', type=int,
                    default=16,
                    help='Batch size for training and validation')
parser.add_argument('-o', '--output_path', type=str,
                    default="./output/",
                    help='Path to directory for saving outputs.')

args = parser.parse_args()

output_path = os.path.join(args.output_path, args.model, datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))

generatorTrain = VVAD_LRS3(path=args.data_path, split="train")
generatorVal = VVAD_LRS3(path=args.data_path, split="val")

datasetTrain = Dataset.from_generator(generatorTrain, output_signature=(tf.TensorSpec(shape=(38, 96, 96, 3)), tf.TensorSpec(shape=(), dtype=tf.int8)))
datasetVal = Dataset.from_generator(generatorVal, output_signature=(tf.TensorSpec(shape=(38, 96, 96, 3)), tf.TensorSpec(shape=(), dtype=tf.int8)))

# Add length of dataset. This needs to be manually set because we use from generator.
datasetTrain = datasetTrain.apply(
    tf.data.experimental.assert_cardinality(len(generatorTrain))
)
datasetVal = datasetVal.apply(
    tf.data.experimental.assert_cardinality(len(generatorVal))
)

datasetTrain = datasetTrain.batch(args.batch_size)
datasetVal = datasetVal.batch(args.batch_size)

n_batches_per_epoch = len(datasetTrain) / args.batch_size
n_batches_per_epoch = math.ceil(n_batches_per_epoch)

model = None

# Python 3.8 does not support switch case statements :(
if args.model == "VVAD_LRS3":
  model = VVAD_LRS3_LSTM()

  # TODO UserWarning: "`binary_crossentropy` received `from_logits=True`, but the `output` argument was produced by a Sigmoid activation and thus does not represent logits. Was this intended?
  #   output, from_logits = _get_logits(
  loss = BinaryCrossentropy()
  optimizer = SGD()

  model.compile(loss=loss, optimizer=optimizer, metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5), 'TrueNegatives', 'TruePositives', 'FalseNegatives', 'FalsePositives'])
elif args.model == "CNN2Plus1D":
  model = CNN2Plus1D()

  loss = BinaryCrossentropy(from_logits=False)  # Alternative for two label Classifications: Hinge Loss or Squared Hinge Loss
  optimizer = Adam(learning_rate=0.0001)

  model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
elif args.model == 'MoViNets':
    # model = MoViNets()
    raise NotImplementedError
elif args.model == 'ViViT':
    # model = ViViT()
    raise NotImplementedError
else:
  raise Exception("Model name not found")

# Checkpoint callback that saves the weights of the network every 20 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(output_path, "checkpoints/weights-{epoch:02d}.hdf5"),
    verbose=1,
    save_weights_only=True,
    # TODO only for testing save_freq=10*n_batches_per_epoch
)

tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join(output_path, 'tensorboard_logs'),  # os.path.join(args.output_path, args.model, 'tensorboard_logs'),
    # don't think I need weight histogramms histogram_freq=1,
    update_freq='epoch'
)

csv_callback = callbacks.CSVLogger(filename=os.path.join(output_path, 'outputs_csv.log'))

# tracker = OfflineEmissionsTracker(project_name="VVAD", experiment_id=args.model, country_iso_code="DEU", output_dir=output_path, output_file="codecarbon", tracking_mode="process") # gpu_ids=[0,1,2,3], on_csv_write="append/update"
# tracker.start()

model.fit(x = datasetTrain,
                    epochs = 2,
                    callbacks=[cp_callback, tb_callback, csv_callback],
                    validation_data = datasetVal)

# tracker.stop()