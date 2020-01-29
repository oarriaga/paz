import os
import argparse
from tensorflow import keras

from paz.core import ProcessingSequencer

from datasets import MNIST, CIFAR10
from models import CNN
from pipelines import ImageAugmentation
from callbacks import Evaluator

splits = ['train', 'test']

description = 'Unsupervised difficulty estimation for classification'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-p', '--save_path', default='evaluations', type=str,
                    help='Path for saving evaluations')
parser.add_argument('-m', '--model', default='CNN', type=str,
                    choices=['CNN'])
parser.add_argument('-d', '--dataset', default='MNIST', type=str,
                    choices=['MNIST', 'CIFAR10'])
parser.add_argument('-b', '--batch_size', default=32, type=int,
                    help='Batch size used during optimization')
parser.add_argument('-e', '--epochs', default=20, type=int,
                    help='Number of epochs before finishing')
args = parser.parse_args()

name_to_manager = {'MNIST': MNIST, 'CIFAR10': CIFAR10}
data_managers, datasets = {}, {}
for split in splits:
    data_manager = name_to_manager[args.dataset](split)
    data = data_manager.load()
    data_managers[split], datasets[split] = data_manager, data

sequencers = {}
for split in splits:
    manager = data_managers[split]
    processor = ImageAugmentation(manager.image_size[0], manager.num_classes)
    sequencer = ProcessingSequencer(processor, args.batch_size, data)
    sequencers[split] = sequencer

name_to_model = {'CNN': CNN}
Model = name_to_model[args.model]
model = Model(sequencers['train'].processor.input_shapes[0],
              data_managers['train'].num_classes)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

evaluations_filepath = os.path.join(args.save_path, args.dataset + '.hdf5')
evaluators = [keras.losses.CategoricalCrossentropy(reduction='none')]
evaluate = Evaluator(sequencers['test'], 'label', evaluators,
                     args.epochs, evaluations_filepath)

model.fit_generator(
    sequencers['train'],
    steps_per_epoch=10,
    epochs=args.epochs,
    callbacks=[evaluate],
    verbose=1)
