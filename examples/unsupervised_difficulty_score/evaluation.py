import argparse
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt

from datasets import MNIST, CIFAR10
from paz.datasets import FERPlus


description = 'Unsupervised difficulty estimation evaluation'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-p', '--save_path', default='experiments', type=str,
                    help='Path for saving evaluations')
parser.add_argument('-d', '--dataset', default='MNIST', type=str,
                    choices=['MNIST', 'CIFAR10', 'FERPlus'])
parser.add_argument('-e', '--evaluation', default='categorical_crossentropy',
                    type=str, choices=['categorical_crossentropy'])
parser.add_argument('-m', '--model', default='CNN-KERAS-A',
                    type=str, choices=['CNN-KERAS-A', 'CNN-KERAS-B',
                                       'RESNET-V2', 'XCEPTION-MINI'])
parser.add_argument('-k', '--top_k', default=10, type=int,
                    help='top K values to be plotted')
parser.add_argument('-l', '--run_label', default='RUN_00', type=str,
                    help='Label used to distinguish between different runs')
parser.add_argument('-s', '--evaluation_splits', nargs='+', type=str,
                    default=['test'], help='Splits used for evaluation')
parser.add_argument('-r', '--data_path', type=str,
                    default=os.path.join(
                        os.path.expanduser('~'), '.keras/paz/datasets/'),
                    help='Default root data path')
args = parser.parse_args()


def make_mosaic(images, titles=None, rows=1):
    assert((titles is None) or (len(images) == len(titles)))
    num_images = len(images)
    figure = plt.figure()
    for image_arg, (image, title) in enumerate(zip(images, titles)):
        cols = np.ceil(num_images / float(rows))
        axis = figure.add_subplot(rows, cols, image_arg + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        plt.axis('off')
        if titles is not None:
            axis.set_title(title)
    figure.set_size_inches(np.array(figure.get_size_inches()) * num_images)
    return figure


name_to_manager = {'MNIST': MNIST, 'CIFAR10': CIFAR10, 'FERPlus': FERPlus}
data_managers, datasets = {}, {}
for split in args.evaluation_splits:
    data_path = os.path.join(args.data_path, args.dataset)
    kwargs = {'path': data_path} if args.dataset in ['FERPlus'] else {}
    data_manager = name_to_manager[args.dataset](split=split, **kwargs)
    data_managers[split], datasets[split] = data_manager, data_manager.load()

split = args.evaluation_splits[0]
experiment_label = '_'.join([args.dataset, args.model, args.run_label])
evaluation_label = '_'.join([args.evaluation, split + '.hdf5'])
filename = os.path.join(args.save_path, experiment_label, evaluation_label)

evaluations = h5py.File(filename, 'r')
evaluations = np.asarray(evaluations['evaluations'])
evaluations = evaluations[..., :len(datasets[split]), 0]

action_scores = np.sum(evaluations, axis=0)
sorted_scores = np.argsort(action_scores)

print('Displaying easiest samples')
images, titles = [], []
for arg in sorted_scores[:args.top_k]:
    images.append(datasets[split][arg]['image'])
    action_score = round(action_scores[arg], 2)
    titles.append(str(action_score))

figure = make_mosaic(images, titles, rows=3)
figure.suptitle('Easiest examples', fontsize=16)
plt.show()

print('Displaying easiest samples')
images, titles = [], []
for arg in sorted_scores[-args.top_k:]:
    images.append(datasets[split][arg]['image'])
    action_score = round(action_scores[arg], 2)
    titles.append(str(action_score))


figure = make_mosaic(images, titles, rows=3)
figure.suptitle('Most difficult samples', fontsize=16)
plt.show()
