import os
import argparse

import h5py
import numpy as np

from paz.models.classification import VVAD_LRS3_LSTM, MoViNet


def predict(clip):
    y = model.predict(np.array(clip))

    print(f"label: {y > 0.5}, score: {y}")

description = 'VVAD recognition training'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-p', '--data_path', type=str,
                    default=os.path.join(
                        os.path.expanduser('~'), '.keras/paz/datasets/'),
                    help='Default root data path')

args = parser.parse_args()

# model = pretrained_models.getFaceImageModel()
# model = pretrained_models.getFaceFeatureModel()
# model = pretrained_models.getLipImageModel()
# model = pretrained_models.getLipFeatureModel()
# model = VVAD_LRS3_LSTM(weights='VVAD_LRS3_LSTM')
model = MoViNet(weights='MoViNets')

dataset = h5py.File("/media/cedric/SpeedData/Datasets/VVAD/vvadlrs3_faceImages_small.h5", mode='r')

x_train = dataset.get("x_test")
y_train = dataset.get("y_test")

visualized_positive = False
visualized_negative = False
i = 0
pos = 0
neg = 0
# this is not ideal - the dataset is sorted for pos and neg samples. This means at least num_samples/2 + 1 iterations
while (not (visualized_positive and visualized_negative)):
    data = np.array(x_train[i])
    label = bool(y_train[i])
    if not visualized_positive and label:
        print("positive sample")
        predict([x_train[i]])
        pos += 1
        if pos >= 5:
            visualized_positive = True
    if not visualized_negative and not label:
        print("negative sample")
        predict([x_train[i]])
        neg += 1
        if neg >= 5:
            visualized_negative = True
    i += 1