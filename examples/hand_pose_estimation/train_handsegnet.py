import os
import json
import argparse
from datetime import datetime

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalCrossentropy

from paz.abstract import ProcessingSequence
from pipelines import AugmentHandSegmentation
from HandPoseEstimation import Hand_Segmentation_Net
from hand_keypoints_loader import RenderedHandLoader
from utils import load_pretrained_weights

description = 'Training script for semantic segmentation'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--dataset_path', type=str, help='Path to dataset')
parser.add_argument('-p', '--save_path', default='experiments', type=str,
                    help='Path for saving evaluations')
parser.add_argument('-d', '--dataset', default='RHD', type=str,
                    choices=['RHD'])
parser.add_argument('-b', '--batch_size', default=5, type=int,
                    help='Batch size used during optimization')
parser.add_argument('-e', '--epochs', default=100, type=int,
                    help='Number of epochs before finishing')
parser.add_argument('-o', '--stop_patience', default=5, type=int,
                    help='Early stop patience')
parser.add_argument('-u', '--reduce_patience', default=2, type=int,
                    help='Reduce learning rate patience')
parser.add_argument('-l', '--run_label', default='RUN_00', type=str,
                    help='Label used to distinguish between different runs')
parser.add_argument('-s', '--evaluation_splits', nargs='+', type=str,
                    default=['test'], help='Splits used for evaluation')
parser.add_argument('-v', '--validation_split', default='val', type=str,
                    help='Split used for validation')
parser.add_argument('-t', '--time', type=str,
                    default=datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
parser.add_argument('-a', '--activation', type=str, default='softmax',
                    help='Final activation function')
parser.add_argument('-z', '--image_size', default=320, type=int,
                    help='Image size. Value is applied to height and width')
parser.add_argument('-w', '--load_pretrained_weights', default=True, type=bool,
                    help='If True, load pre-trained weights')
parser.add_argument('-wp', '--pretrained_weights_path',
                    default='./person_net.ckpt.meta', type=str,
                    help='Path to pre-trained weights')

args = parser.parse_args()

model = Hand_Segmentation_Net()
loss = CategoricalCrossentropy(from_logits=True)

splits = ['train'] + args.validation_split

name_to_manager = {'RHD': RenderedHandLoader}

data_managers, datasets = {}, {}
for split in splits:
    args_manager = [args.dataset_path]
    data_manager = name_to_manager[args.dataset](*args_manager)
    data_managers[split] = data_manager
    datasets[split] = data_manager.load_data()

# instantiating sequencers
sequencers = {}
for split in splits:
    data_manager = data_managers[split]
    image_shape = (args.image_size, args.image_size)
    processor = AugmentHandSegmentation(image_shape)
    sequencers[split] = ProcessingSequence(
        processor, args.batch_size, datasets[split])

model = Hand_Segmentation_Net()
loss = CategoricalCrossentropy(from_logits=True)

model.compile(loss=loss, optimizer=Adam(), metrics=['mean_squared_error'])

if args.load_pretrained_weights:
    model = load_pretrained_weights(args.pretrained_weights_path, model=model,
                                    num_layers=16)

# creating directory for experiment
callbacks = []
experiment_label = '_'.join([args.dataset, model.name, args.run_label])
experiment_path = os.path.join(args.save_path, experiment_label)
if not os.path.exists(experiment_path):
    os.makedirs(experiment_path)

# setting additional callbacks
log = CSVLogger(os.path.join(experiment_path, 'optimization.log'))
stop = EarlyStopping(patience=args.stop_patience)
plateau = ReduceLROnPlateau(patience=args.reduce_patience)
save_filename = os.path.join(experiment_path, 'model.hdf5')
save = ModelCheckpoint(save_filename, save_best_only=True)
callbacks.extend([log, stop, save, plateau])

# saving hyper-parameters and model summary
with open(os.path.join(experiment_path, 'hyperparameters.json'), 'w') as filer:
    json.dump(args.__dict__, filer, indent=4)
with open(os.path.join(experiment_path, 'model_summary.txt'), 'w') as filer:
    model.summary(print_fn=lambda x: filer.write(x + '\n'))

# starting optimization
model.fit(
    sequencers['train'],
    epochs=args.epochs,
    validation_data=sequencers[args.validation_split],
    callbacks=callbacks,
    verbose=1,
    workers=1,
    use_multiprocessing=False)

# saving using model tf
save_filename = os.path.join(experiment_path, 'model.tf')
model.save_weights(save_filename, save_format='tf')
