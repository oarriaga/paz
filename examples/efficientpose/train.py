import argparse
import os
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from paz.abstract import ProcessingSequence
from paz.optimization import MultiBoxLoss
from paz.optimization.callbacks import LearningRateScheduler
from paz.processors import TRAIN, VAL
from linemod import LINEMOD
from pose import AugmentPose, EFFICIENTPOSEA
from losses import MultiPoseLoss


gpus = tf.config.experimental.list_physical_devices('GPU')


description = 'Training script for single-shot object detection models'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-bs', '--batch_size', default=16, type=int,
                    help='Batch size for training')
parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float,
                    help='Initial learning rate for SGD')
parser.add_argument('-m', '--momentum', default=0.9, type=float,
                    help='Momentum for SGD')
parser.add_argument('-g', '--gamma_decay', default=0.1, type=float,
                    help='Gamma decay for learning rate scheduler')
parser.add_argument('-e', '--num_epochs', default=1000, type=int,
                    help='Maximum number of epochs before finishing')
parser.add_argument('-iou', '--AP_IOU', default=0.5, type=float,
                    help='Average precision IOU used for evaluation')
parser.add_argument('-sp', '--save_path', default='trained_models/',
                    type=str, help='Path for writing model weights and logs')
parser.add_argument('-dp', '--data_path', default='Linemod_preprocessed/',
                    type=str, help='Path for writing model weights and logs')
parser.add_argument('-id', '--object_id', default='08',
                    type=str, help='ID of the object to train')
parser.add_argument('-se', '--scheduled_epochs', nargs='+', type=int,
                    default=[1000, 1250], help='Epoch learning rate reduction')
parser.add_argument('-mp', '--multiprocessing', default=False, type=bool,
                    help='Select True for multiprocessing')
parser.add_argument('-w', '--workers', default=1, type=int,
                    help='Number of workers used for optimization')
args = parser.parse_args()

optimizer = Adam(learning_rate=args.learning_rate, clipnorm=0.001)

data_splits = ['train', 'test']
data_names = ['LINEMOD', 'LINEMOD']

# loading datasets
data_managers, datasets, evaluation_data_managers = [], [], []
for data_name, data_split in zip(data_names, data_splits):
    data_manager = LINEMOD(args.data_path, args.object_id,
                           data_split, name=data_name)
    data_managers.append(data_manager)
    datasets.append(data_manager.load_data())
    if data_split == 'test':
        eval_data_manager = LINEMOD(
            args.data_path, args.object_id, data_split,
            name=data_name, evaluate=True)
        evaluation_data_managers.append(eval_data_manager)

# instantiating model
num_classes = data_managers[0].num_classes
model = EFFICIENTPOSEA(num_classes, base_weights='COCO', head_weights=None)
model.summary()

# Instantiating loss and metrics
box_loss = MultiBoxLoss()
transformation_loss = MultiPoseLoss(args.object_id, model.translation_priors,
                                    args.data_path)
loss = {'boxes': box_loss.compute_loss,
        'transformation': transformation_loss.compute_loss}
loss_weights = {'boxes': 1.0,
                'transformation': 0.02}
metrics = {'boxes': [box_loss.localization,
                     box_loss.positive_classification,
                     box_loss.negative_classification]}
model.compile(optimizer=optimizer, loss=loss,
              metrics=metrics, loss_weights=loss_weights)

# setting data augmentation pipeline
augmentators = []
for split in [TRAIN, VAL]:
    augmentator = AugmentPose(model, split, size=512, num_classes=num_classes)
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
    initial_epoch=0,
    steps_per_epoch=-1,
    verbose=1,
    callbacks=[checkpoint, log, schedule],
    validation_data=sequencers[1],
    use_multiprocessing=args.multiprocessing,
    workers=args.workers)
