import os
import glob
import argparse
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()
# import tensorflow as tf
# tf.compat.v1.experimental.output_all_intermediates(True)
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from paz.optimization.callbacks import LearningRateScheduler
from pipelines import AugmentDetection
from paz.models import SSD300
from data_manager import CSVLoader
from paz.optimization import MultiBoxLoss
# from paz.abstract import ProcessingSequence
from sequencer import ProcessingSequence
from paz.optimization.callbacks import EvaluateMAP
from paz.pipelines import DetectSingleShot
from paz.processors import TRAIN, VAL

description = 'Training script for single-shot object detection models'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-bs', '--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('-et', '--evaluation_period', default=10, type=int,
                    help='evaluation frequency')
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float,
                    help='Initial learning rate for SGD')
parser.add_argument('-m', '--momentum', default=0.9, type=float,
                    help='Momentum for SGD')
parser.add_argument('-g', '--gamma_decay', default=0.1, type=float,
                    help='Gamma decay for learning rate scheduler')
parser.add_argument('-e', '--num_epochs', default=240, type=int,
                    help='Maximum number of epochs before finishing')
parser.add_argument('-iou', '--AP_IOU', default=0.5, type=float,
                    help='Average precision IOU used for evaluation')
parser.add_argument('-sp', '--save_path', default='trained_models/',
                    type=str, help='Path for writing model weights and logs')
parser.add_argument('-dp', '--data_path', default='VOCdevkit/',
                    type=str, help='Path for writing model weights and logs')
parser.add_argument('-bk', '--bkg_path',
                    default=os.path.join(
                        os.path.expanduser('~'),
                        '.keras/paz/datasets/voc-backgrounds/'),
                    type=str, help='Path for writing model weights and logs')
parser.add_argument('-se', '--scheduled_epochs', nargs='+', type=int,
                    default=[110, 152], help='Epoch learning rate reduction')
parser.add_argument('-mp', '--multiprocessing', default=False, type=bool,
                    help='Select True for multiprocessing')
parser.add_argument('-w', '--workers', default=1, type=int,
                    help='Number of workers used for optimization')
args = parser.parse_args()

optimizer = SGD(args.learning_rate, args.momentum)

data_splits = ['train']
# loading datasets
data_managers, datasets = [], []
for data_split in zip(data_splits):
    path = 'datasets/solar_panel/BoundingBox.txt'
    class_names = ['background', 'solar_panel']
    data_manager = CSVLoader(path, class_names, split=data_split)
    data_managers.append(data_manager)
    datasets.append(data_manager.load_data())

# instantiating model
num_classes = data_managers[0].num_classes
model = SSD300(num_classes, base_weights='VOC', head_weights=None,
               trainable_base=False)
model.summary()

# Instantiating loss and metrics
loss = MultiBoxLoss()
metrics = {'boxes': [loss.localization,
                     loss.positive_classification,
                     loss.negative_classification]}
model.compile(optimizer, loss.compute_loss, metrics)

# setting data augmentation pipeline
bkg_paths = glob.glob(args.bkg_path + '*.png')
if len(bkg_paths) == 0:
    raise ValueError('No background png files were found in', args.bkg_path)
augmentators = []
for split in [TRAIN, VAL]:
    augmentator = AugmentDetection(model.prior_boxes, bkg_paths, split)
    augmentators.append(augmentator)

# setting sequencers
num_steps = 1000
sequencers = []
for data, augmentator in zip(datasets, augmentators):
    sequencer = ProcessingSequence(
        augmentator, args.batch_size, data, num_steps)
    sequencers.append(sequencer)

# setting callbacks
model_path = os.path.join(args.save_path, model.name)
if not os.path.exists(model_path):
    os.makedirs(model_path)
log = CSVLogger(os.path.join(model_path, model.name + '-optimization.log'))
save_path = os.path.join(model_path, 'weights.{epoch:02d}-{loss:.2f}.hdf5')
checkpoint = ModelCheckpoint(
    save_path, 'loss', verbose=1, save_weights_only=True)
schedule = LearningRateScheduler(
    args.learning_rate, args.gamma_decay, args.scheduled_epochs)

# training
model.fit_generator(
    sequencers[0],
    epochs=args.num_epochs,
    # steps_per_epoch=100,
    verbose=1,
    callbacks=[checkpoint, log, schedule],
    # validation_data=sequencers[1],
    use_multiprocessing=args.multiprocessing,
    workers=args.workers)
