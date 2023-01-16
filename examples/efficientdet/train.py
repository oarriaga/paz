import argparse
import os
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from paz.abstract import ProcessingSequence
from paz.datasets import VOC
from paz.optimization import MultiBoxLoss
from paz.optimization.callbacks import LearningRateScheduler
from paz.pipelines import AugmentDetection
from paz.processors import TRAIN, VAL
from efficientdet import EFFICIENTDETD0

gpus = tf.config.experimental.list_physical_devices('GPU')

description = 'Training script for single-shot object detection models'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-bs', '--batch_size', default=128, type=int,
                    help='Batch size for training')
parser.add_argument('-lr', '--learning_rate', default=0.08, type=float,
                    help='Initial learning rate for SGD')
parser.add_argument('-m', '--momentum', default=0.9, type=float,
                    help='Momentum for SGD')
parser.add_argument('-g', '--gamma_decay', default=0.1, type=float,
                    help='Gamma decay for learning rate scheduler')
parser.add_argument('-e', '--num_epochs', default=300, type=int,
                    help='Maximum number of epochs before finishing')
parser.add_argument('-iou', '--AP_IOU', default=0.5, type=float,
                    help='Average precision IOU used for evaluation')
parser.add_argument('-sp', '--save_path', default='trained_models/',
                    type=str, help='Path for writing model weights and logs')
parser.add_argument('-dp', '--data_path', default='VOCdevkit/',
                    type=str, help='Path for writing model weights and logs')
parser.add_argument('-se', '--scheduled_epochs', nargs='+', type=int,
                    default=[200, 250], help='Epoch learning rate reduction')
parser.add_argument('-mp', '--multiprocessing', default=False, type=bool,
                    help='Select True for multiprocessing')
parser.add_argument('-w', '--workers', default=1, type=int,
                    help='Number of workers used for optimization')
args = parser.parse_args()

optimizer = SGD(args.learning_rate, args.momentum)

data_splits = [['trainval', 'trainval'], 'test']
data_names = [['VOC2007', 'VOC2012'], 'VOC2007']

# loading datasets
data_managers, datasets, evaluation_data_managers = [], [], []
for data_name, data_split in zip(data_names, data_splits):
    data_manager = VOC(args.data_path, data_split, name=data_name)
    data_managers.append(data_manager)
    datasets.append(data_manager.load_data())
    if data_split == 'test':
        eval_data_manager = VOC(
            args.data_path, data_split, name=data_name, evaluate=True)
        evaluation_data_managers.append(eval_data_manager)

# instantiating model
num_classes = data_managers[0].num_classes
model = EFFICIENTDETD0(num_classes, base_weights='COCO', head_weights=None)
model.summary()

# Instantiating loss and metrics
loss = MultiBoxLoss()
metrics = {'boxes': [loss.localization,
                     loss.positive_classification,
                     loss.negative_classification]}
model.compile(optimizer, loss.compute_loss, metrics)

# setting data augmentation pipeline
augmentators = []
for split in [TRAIN, VAL]:
    augmentator = AugmentDetection(model.prior_boxes, split, size=512)
    augmentators.append(augmentator)

# setting sequencers
sequencers = []
for data, augmentator in zip(datasets, augmentators):
    sequencer = ProcessingSequence(augmentator, args.batch_size, data)
    sequencers.append(sequencer)

# setting callbacks
model_path = os.path.join(args.save_path, model.name)
if not os.path.exists(model_path):
    os.makedirs(model_path)
log = CSVLogger(os.path.join(model_path, model.name + '-optimization.log'))
save_path = os.path.join(model_path, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
checkpoint = ModelCheckpoint(save_path, verbose=1, save_weights_only=True)
schedule = LearningRateScheduler(
    args.learning_rate, args.gamma_decay, args.scheduled_epochs)

# training
model.fit(
    sequencers[0],
    epochs=args.num_epochs,
    verbose=1,
    callbacks=[checkpoint, log, schedule],
    validation_data=sequencers[1],
    use_multiprocessing=args.multiprocessing,
    workers=args.workers)
