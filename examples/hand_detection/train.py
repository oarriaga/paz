import os
import json
import argparse
from datetime import datetime

# from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau)

import paz.processors as pr
from paz.abstract import ProcessingSequence
from paz.pipelines import AugmentDetection
from paz.optimization import MultiBoxLoss

from open_images import OpenImagesV6
from paz.models import SSD300

root_path = os.path.expanduser('~')
DEFAULT_DATA_PATH = os.path.join(root_path, 'hand_dataset/hand_dataset/')

description = 'Training script for single-shot object detection models'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--evaluation_frequency', default=10, type=int,
                    help='evaluation frequency')
parser.add_argument('--stop_patience', default=5, type=int,
                    help='Early stop patience')
parser.add_argument('--reduce_patience', default=2, type=int,
                    help='Reduce learning rate patience')
parser.add_argument('--learning_rate', default=0.0001, type=float,
                    help='Initial learning rate for SGD')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum for SGD')
parser.add_argument('--gamma_decay', default=0.1, type=float,
                    help='Gamma decay for learning rate scheduler')
parser.add_argument('--num_epochs', default=240, type=int,
                    help='Maximum number of epochs before finishing')
parser.add_argument('--AP_IOU', default=0.5, type=float,
                    help='Average precision IOU used for evaluation')
parser.add_argument('--save_path', default='experiments',
                    type=str, help='Path for writing model weights and logs')
parser.add_argument('--data_path', default=DEFAULT_DATA_PATH,
                    type=str, help='Path for writing model weights and logs')
parser.add_argument('--scheduled_epochs', nargs='+', type=int,
                    default=[110, 152], help='Epoch learning rate reduction')
parser.add_argument('--run_label', default='RUN_00', type=str,
                    help='Label used to distinguish between different runs')
args = parser.parse_args()


# loading datasets
"""
data_managers, datasets = [], []
for split in [pr.TRAIN, pr.VAL, pr.TEST]:
    data_manager = HandDataset(args.data_path, split)
    data = data_manager.load_data()
    data_managers.append(data_manager)
    datasets.append(data)

from egohand_dataset import EgoHands
path = os.path.join(root_path, 'Downloads/egohands/_LABELLED_SAMPLES/')
data_manager = EgoHands(path)
ego_data = data_manager.load_data()
datasets[0].extend(ego_data)
"""

path = os.path.join(root_path, '/home/octavio/Datasets/fiftyone/open-images-v6/')
data_managers, datasets = [], []
for split in [pr.TRAIN, pr.VAL, pr.TEST]:
    data_manager = OpenImagesV6(path, split, ['background', 'Human hand'])
    data = data_manager.load_data()
    data_managers.append(data_manager)
    datasets.append(data)


# instantiating model
num_classes = data_managers[0].num_classes
from model import SSD512Custom

model = SSD512Custom(num_classes, trainable_base=True)
"""
model = SSD300(num_classes, base_weights='VOC', head_weights=None,
               trainable_base=True)
model.load_weights('experiments/SSD300_RUN_00_10-06-2022_16-55-40/model_weights.hdf5')
"""
size = model.input_shape[1]


# Instantiating loss and metrics
# optimizer = SGD(args.learning_rate, args.momentum)
optimizer = Adam(args.learning_rate, amsgrad=True)
loss = MultiBoxLoss()
metrics = {'boxes': [loss.localization,
                     loss.positive_classification,
                     loss.negative_classification]}
model.compile(optimizer, loss.compute_loss, metrics)

# build augmentation pipelines
augmentators = []
for split in [pr.TRAIN, pr.VAL]:
    augmentator = AugmentDetection(model.prior_boxes, split, num_classes, size)
    augmentators.append(augmentator)

# EXPERIMENTAL: removes RandomSampleCrop
augmentators[0].augment_boxes.pop(2)

# build sequencers
sequencers = []
for data, processor in zip(datasets, augmentators):
    sequencer = ProcessingSequence(processor, args.batch_size, data)
    sequencers.append(sequencer)

# saving hyper-parameters and model summary
current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
experiment_label = '_'.join([model.name, args.run_label, current_time])
experiment_path = os.path.join(args.save_path, experiment_label)
if not os.path.exists(experiment_path):
    os.makedirs(experiment_path)
with open(os.path.join(experiment_path, 'hyperparameters.json'), 'w') as filer:
    json.dump(args.__dict__, filer, indent=4)
with open(os.path.join(experiment_path, 'model_summary.txt'), 'w') as filer:
    model.summary(print_fn=lambda x: filer.write(x + '\n'))

# setting additional callbacks
log = CSVLogger(os.path.join(experiment_path, 'optimization.log'))
stop = EarlyStopping(patience=args.stop_patience, verbose=1)
plateau = ReduceLROnPlateau(patience=args.reduce_patience, verbose=1)
save_name = os.path.join(experiment_path, 'model_weights.hdf5')
save = ModelCheckpoint(save_name, verbose=1, save_best_only=True,
                       save_weights_only=True)

# training
model.fit(
    sequencers[0],
    epochs=args.num_epochs,
    verbose=1,
    callbacks=[log, stop, plateau, save],
    validation_data=sequencers[1],
    use_multiprocessing=True,
    workers=6)
