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

from paz.abstract import ProcessingSequence
from paz.optimization import DiceLoss, FocalLoss, JaccardLoss
from paz.models import UNET_VGG16, UNET_VGG19, UNET_RESNET50
from paz.datasets import CityScapes

from pipelines import PreprocessSegmentationIds


description = 'Training script for semantic segmentation'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--label_path', type=str, help='Path to labels')
parser.add_argument('--image_path', type=str, help='Path to images')
parser.add_argument('-p', '--save_path', default='experiments', type=str,
                    help='Path for saving evaluations')
parser.add_argument('-m', '--model', default='UNET_VGG16', type=str,
                    choices=['UNET_VGG16', 'UNET_VGG19', 'UNET_RESNET50'])
parser.add_argument('-d', '--dataset', default='CityScapes', type=str,
                    choices=['CityScapes'])
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
parser.add_argument('-z', '--image_size', default=128, type=int,
                    help='Image size. Value is applied to height and width')
parser.add_argument('-f', '--freeze_backbone', default=True, type=bool,
                    help='If True backbone in UNET is frozen.')
parser.add_argument('-x', '--loss', default='dice', type=str,
                    choices=['dice', 'jaccard', 'focal', 'all'])
args = parser.parse_args()


# build split names
splits = ['train'] + args.evaluation_splits
if args.validation_split not in splits:
    splits = splits + [args.validation_split]

# loading data and instantiating data managers
name_to_manager = {'CityScapes': CityScapes}

# loading splits
data_managers, datasets = {}, {}
for split in splits:
    args_manager = [args.image_path, args.label_path, split]
    data_manager = name_to_manager[args.dataset](*args_manager)
    data_managers[split] = data_manager
    datasets[split] = data_manager.load_data()

# instantiating sequencers
sequencers = {}
for split in splits:
    data_manager = data_managers[split]
    num_classes = data_manager.num_classes
    image_shape = (args.image_size, args.image_size)
    processor = PreprocessSegmentationIds(image_shape, num_classes)
    sequencers[split] = ProcessingSequence(
        processor, args.batch_size, datasets[split])

# instantiating model
name_to_model = {'UNET_VGG16': UNET_VGG16, 'UNET_VGG19': UNET_VGG19,
                 'UNET_RESNET50': UNET_RESNET50}
Model = name_to_model[args.model]
num_classes = data_managers['train'].num_classes
input_shape = processor.processors[-1].inputs_info[0]['input_1']
print(num_classes, input_shape)
model = Model(num_classes, input_shape, freeze_backbone=args.freeze_backbone,
              activation=args.activation)

# building loss function
if args.loss == 'all':
    loss = DiceLoss() + JaccardLoss() + FocalLoss()
else:
    get_loss = {'dice': DiceLoss, 'jaccard': JaccardLoss, 'focal': FocalLoss}
    loss = get_loss[args.loss]()

# compiling model
model.compile(loss=loss,
              optimizer=Adam(),
              metrics=['mean_squared_error'])

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
