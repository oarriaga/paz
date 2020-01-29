import os
import argparse
from tensorflow import keras

from paz.core import ProcessingSequencer
from paz.datasets import FERPlus

from datasets import MNIST, CIFAR10
from models import CNN
from pipelines import ImageAugmentation
from callbacks import Evaluator


description = 'Unsupervised difficulty estimation for classification'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-p', '--save_path', default='evaluations', type=str,
                    help='Path for saving evaluations')
parser.add_argument('-m', '--model', default='CNN', type=str,
                    choices=['CNN'])
parser.add_argument('-d', '--dataset', default='MNIST', type=str,
                    choices=['MNIST', 'CIFAR10', 'FERPlus'])
parser.add_argument('-b', '--batch_size', default=32, type=int,
                    help='Batch size used during optimization')
parser.add_argument('-e', '--epochs', default=20, type=int,
                    help='Number of epochs before finishing')
parser.add_argument('-l', '--experiment_label', default='00', type=str,
                    help='Label used to distinguish between different runs')
parser.add_argument('-s', '--evaluation_splits', nargs='+', type=str,
                    default=['test'], help='Splits used for evaluation')
parser.add_argument('-r', '--data_path', type=str,
                    default=os.path.join(
                        os.path.expanduser('~'), '.keras/paz/datasets/'),
                    help='Default root data path')
args = parser.parse_args()

splits = ['train'] + args.evaluation_splits
name_to_manager = {'MNIST': MNIST, 'CIFAR10': CIFAR10, 'FERPlus': FERPlus}
data_managers, datasets = {}, {}

for split in splits:
    data_path = os.path.join(args.data_path, args.dataset)
    kwargs = {'path': data_path} if args.dataset in ['FERPlus'] else {}
    print(kwargs)
    data_manager = name_to_manager[args.dataset](split=split, **kwargs)
    data = data_manager.load()
    data_managers[split], datasets[split] = data_manager, data

sequencers = {}
for split in splits:
    data_manager = data_managers[split]
    size, num_classes = data_manager.image_size[0], data_manager.num_classes
    grayscale = True if args.dataset in ['MNIST', 'FERPlus'] else False
    processor = ImageAugmentation(size, num_classes, grayscale)
    sequencer = ProcessingSequencer(processor, args.batch_size, data)
    sequencers[split] = sequencer

name_to_model = {'CNN': CNN}
Model = name_to_model[args.model]
model = Model(sequencers['train'].processor.input_shapes[0],
              data_managers['train'].num_classes)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

callbacks = []
for split in args.evaluation_splits:
    filename = '_'.join([args.dataset, split, args.experiment_label])
    evaluations_filepath = os.path.join(args.save_path, filename + '.hdf5')
    evaluators = [keras.losses.CategoricalCrossentropy(reduction='none')]
    callbacks.append(Evaluator(sequencers[split], 'label',
                     evaluators, args.epochs, evaluations_filepath))

model.fit_generator(
    sequencers['train'],
    epochs=args.epochs,
    callbacks=callbacks,
    verbose=1)
