import os
import argparse
import numpy as np
import tensorflow as tf
from paz.backend.image import lincolor
import seaborn as sns
import matplotlib.pyplot as plt


from paz.abstract import ProcessingSequence

from facial_keypoints import FacialKeypoints
from pipelines import AugmentKeypoints
from model import GaussianMixtureModel
from paz.backend.image import show_image
from paz.backend.keypoints import denormalize_keypoints
from processors import draw_circles

description = 'Training script for learning 2D probabilistic keypoints'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-f', '--filters', default=32, type=int,
                    help='Number of filters in convolutional blocks')
parser.add_argument('-b', '--batch_size', default=10, type=int,
                    help='Batch size for training')
parser.add_argument('-nk', '--num_keypoints', default=15, type=int,
                    help='Number of keypoints')
parser.add_argument('-is', '--image_size', default=96, type=int,
                    help='Image size')
parser.add_argument('-vs', '--validation_split', default=0.2, type=float,
                    help='Fraction of the training set used for validation')
parser.add_argument('-s', '--save_path',
                    default=os.path.join(
                        os.path.expanduser('~'), '.keras/paz/models'),
                    type=str, help='Path for writing model weights and logs')
args = parser.parse_args()


# loading training dataset
data_manager = FacialKeypoints('dataset/', 'train')
data = data_manager.load_data()

# split training data-set into train and validation
num_train_samples = int(len(data) * (1 - args.validation_split))
datasets = {'train': data[:num_train_samples],
            'validation': data[num_train_samples:]}

# instantiate keypoint augmentations
kwargs = {'with_partition': True, 'num_keypoints': args.num_keypoints}
processor = AugmentKeypoints('validation', **kwargs)

# creating sequencers
sequence = ProcessingSequence(processor, args.batch_size, data, True)

# instantiate model
batch_shape = (1, args.image_size, args.image_size, 1)
model = GaussianMixtureModel(batch_shape, args.num_keypoints, args.filters)
model.summary()


# making directory for saving model weights and logs
model_name = '_'.join(['FaceKP', model.name, str(args.num_keypoints)])
save_path = os.path.join(args.save_path, model_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)
model_path = os.path.join(save_path, '%s_weights.hdf5' % model_name)

model.load_weights(model_path)
for arg in range(1, 100):
    # arg = 0
    sample = datasets['validation'][arg]
    sample = processor(sample)
    image = sample['inputs']['image']
    image = np.expand_dims(image, axis=0)
    x = np.linspace(-1, 1, 90).astype('float32')
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(4, 4))
    grid = tf.stack(np.meshgrid(x, x), axis=2)
    predictions = model(image)
    keypoints = np.zeros((15, 2))
    colors = lincolor(args.num_keypoints, normalized=True)
    for keypoint_arg, distribution in enumerate(predictions):
        keypoints[keypoint_arg] = distribution.mean()
        probability = distribution.prob(grid).numpy()
        color = colors[keypoint_arg]
        cmap = sns.light_palette(color, input='hsl', as_cmap=True)
        axes[0].contour(probability, cmap=cmap)
    keypoints = denormalize_keypoints(keypoints, 96, 96)
    keypoints = keypoints.astype('int')
    image = (image[0] * 255).astype('uint8')
    image = draw_circles(image, keypoints)
    axes[1].imshow(image[..., 0])
    plt.show()
