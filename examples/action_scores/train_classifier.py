import os
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
import json
import argparse
from datetime import datetime

from tensorflow import keras
from paz.abstract import ProcessingSequence
from paz.datasets import FERPlus

from datasets import MNIST, CIFAR10, FashionMNIST, KuzushijiMNIST, ImageNet64
from datasets import FER
from models import CNN_KERAS_A, CNN_KERAS_B, XCEPTION_MINI, RESNET_V2
from pipelines import ProcessImage
from callbacks import ScalarActionScore, FeatureExtractor


description = 'Unsupervised difficulty estimation for classification'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-p', '--save_path', default='experiments', type=str,
                    help='Path for saving evaluations')
parser.add_argument('-m', '--model', default='CNN-KERAS-A', type=str,
                    choices=['CNN-KERAS-A', 'CNN-KERAS-B',
                             'XCEPTION-MINI', 'RESNET-V2'])
parser.add_argument('-d', '--dataset', default='MNIST', type=str,
                    choices=['MNIST', 'CIFAR10', 'FashionMNIST', 'FER',
                             'KuzushijiMNIST', 'FERPlus', 'ImageNet64'])
parser.add_argument('-b', '--batch_size', default=32, type=int,
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
parser.add_argument('-v', '--validation_split', default='test', type=str,
                    help='Split used for validation')
parser.add_argument('-t', '--time', type=str,
                    default=datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
parser.add_argument('-r', '--data_path', type=str,
                    default=os.path.join(
                        os.path.expanduser('~'), '.keras/paz/datasets/'),
                    help='Default root data path')
args = parser.parse_args()

splits = ['train'] + args.evaluation_splits
if args.validation_split not in splits:
    splits = splits + [args.validation_split]

# loading data and instantiating data managers
name_to_manager = {'MNIST': MNIST,
                   'FERPlus': FERPlus,
                   'FER': FER,
                   'CIFAR10': CIFAR10,
                   'FashionMNIST': FashionMNIST,
                   'KuzushijiMNIST': KuzushijiMNIST,
                   'ImageNet64': ImageNet64}

# checking if manual dataset is required
PAZ_datasets = ['FERPlus', 'ImageNet64', 'FER']
if args.dataset in PAZ_datasets:
    print('WARNING: Manual download required for dataset:', args.dataset)

data_managers, datasets = {}, {}
for split in splits:
    data_path = os.path.join(args.data_path, args.dataset)
    kwargs = {'path': data_path} if args.dataset in PAZ_datasets else {}
    data_manager = name_to_manager[args.dataset](split=split, **kwargs)
    data_managers[split] = data_manager
    datasets[split] = data_manager.load_data()

# instantiating sequencers
sequencers = {}
grayscale_datasets = ['MNIST',
                      'FashionMNIST',
                      'KuzushijiMNIST',
                      'FERPlus', 'FER']
for split in splits:
    data_manager = data_managers[split]
    size, num_classes = data_manager.image_size[0], data_manager.num_classes
    grayscale = True if args.dataset in grayscale_datasets else False
    one_hot_vector = True if args.dataset == 'ImageNet64' else False
    processor = ProcessImage(size, num_classes, grayscale, one_hot_vector)
    sequencers[split] = ProcessingSequence(
        processor, args.batch_size, datasets[split])

# instantiating model
name_to_model = {'RESNET-V2': RESNET_V2,
                 'CNN-KERAS-A': CNN_KERAS_A,
                 'CNN-KERAS-B': CNN_KERAS_B,
                 'XCEPTION-MINI': XCEPTION_MINI}
Model = name_to_model[args.model]
model = Model(processor.processors[-1].inputs_info[0]['image'],
              data_managers['train'].num_classes)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# setting difficulty score callback
experiment_label = '_'.join([args.dataset, model.name, args.run_label])
experiment_path = os.path.join(args.save_path, experiment_label)
callbacks = []
for split in args.evaluation_splits:
    for evaluator in [keras.losses.CategoricalCrossentropy(reduction='none')]:
        filename = '_'.join([evaluator.name, split + '.hdf5'])
        evaluations_filepath = os.path.join(experiment_path, filename)
        evaluators = [keras.losses.CategoricalCrossentropy(reduction='none')]
        callbacks.append(ScalarActionScore(
            sequencers[split], 'label', evaluators,
            args.epochs, evaluations_filepath))

# setting feature extraction callback
model_to_layer = {'CNN-KERAS-A': 'dense',
                  'CNN-KERAS-B': 'dense',
                  'RESNET-V2': 'flatten',
                  'XCEPTION-MINI': 'add_5'}
for split in args.evaluation_splits:
    layer_name = model_to_layer[args.model]
    filename = '_'.join([layer_name, 'features', split + '.hdf5'])
    features_filepath = os.path.join(experiment_path, filename)
    callbacks.append(FeatureExtractor(
        layer_name, sequencers[split], features_filepath))

# setting additional callbacks
log_filename = os.path.join(experiment_path, 'optimization.log')
log = keras.callbacks.CSVLogger(log_filename)
stop = keras.callbacks.EarlyStopping(patience=args.stop_patience)
save_filename = os.path.join(experiment_path, 'model.hdf5')
save = keras.callbacks.ModelCheckpoint(save_filename, save_best_only=True)
plateau = keras.callbacks.ReduceLROnPlateau(patience=args.reduce_patience)
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
    verbose=1)
