import os
import glob
import pickle

import numpy as np
from paz.abstract import Loader
# from paz.backend.image import show_image


class ImageNet(Loader):
    def __init__(self, path, split, image_size, name):
        self.image_size = image_size
        self.query = {'train': 'train_data_batch_*', 'val': 'val_data'}[split]
        labels_path = os.path.join(path, 'imagenet_labels.txt')
        class_names = list(load_arg_to_name(labels_path).values())
        super(ImageNet, self).__init__(path, split, class_names, name)

    def load_data(self):
        fullpaths = get_path_queries(self.path, self.query)
        images, labels = [], []
        for fullpath in fullpaths:
            batch_images, batch_labels = load_batch(fullpath)
            batch_images = transform_shape(batch_images, self.image_size[0])
            batch_labels = shift_back_label_index(batch_labels)
            images.append(batch_images)
            labels.append(batch_labels)
        images = np.concatenate(images, axis=0)
        labels = np.concatenate(labels, axis=0)
        data = self.arrays_to_samples(images, labels)
        return data

    def arrays_to_samples(self, images, labels):
        data = []
        for image, label in zip(images, labels):
            data.append({'image': image, 'label': label})
        return data


class ImageNet64(ImageNet):
    def __init__(self, path, split='train'):
        super(ImageNet64, self).__init__(path, split, (64, 64), 'ImageNet64')


def unpickle(filename):
    with open(filename, 'rb') as filedata:
        dict = pickle.load(filedata)
    return dict


def split_channels(image, image_size):
    image_size_squared = image_size * image_size
    R_channel = image[:, :image_size_squared]
    G_channel = image[:, image_size_squared: (2 * image_size_squared)]
    B_channel = image[:, (2 * image_size_squared):]
    return [R_channel, G_channel, B_channel]


def concatenate_channels(RGB_channels, image_size):
    R_channel, G_channel, B_channel = RGB_channels
    R_channel = R_channel.reshape((-1, image_size, image_size, 1))
    G_channel = G_channel.reshape((-1, image_size, image_size, 1))
    B_channel = B_channel.reshape((-1, image_size, image_size, 1))
    image = np.concatenate([R_channel, G_channel, B_channel], axis=-1)
    return image


def transform_shape(images, image_size):
    RGB_channels = split_channels(images, image_size)
    images = concatenate_channels(RGB_channels, image_size)
    return images


def load_batch(fullpath):
    dataset = unpickle(fullpath)
    images, labels = dataset['data'], dataset['labels']
    return images, labels


def shift_back_label_index(labels):
    labels = np.asarray(labels)
    labels = labels - 1
    return labels


def get_path_queries(directory, filename):
    path_query = os.path.join(directory, filename)
    paths = glob.glob(path_query)
    if len(paths) != 1:
        paths.sort(key=path_to_number)
    return paths


def path_to_number(fullpath):
    directories = fullpath.split('/')
    splits = directories[-1].split('_')
    return int(splits[-1])


def load_arg_to_name(fullpath):
    labels = np.genfromtxt(fullpath, dtype=str)
    args, names = labels[:, 1], labels[:, 2]
    args = args.astype(int) - 1
    arg_to_name = dict(zip(args, names))
    return arg_to_name


"""
path = '/home/octavio/.keras/paz/datasets/ImageNet64/'
imagenet_labels = 'imagenet_labels.txt'
data_manager = ImageNet64(path, 'train')
dataset = data_manager.load_data()

for arg in range(100):
    random_arg = np.random.randint(0, len(dataset))
    sample = dataset[random_arg]
    image, label = sample['image'], sample['label']
    name = data_manager.class_names[label]
    print(name)
    show_image(image, name)
"""
