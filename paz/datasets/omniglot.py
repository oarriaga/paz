import os
import glob
import numpy as np

from tensorflow.keras.utils import Sequence, get_file

from ..backend.image import load_image, resize_image, make_mosaic
from ..abstract import Loader
from ..utils.documentation import docstring


def download(split):
    """Downloads omniglot dataset from original repository source.

    # Arguments:
        split: String indicating dataset split i.e. `train` or `test`.

    # Returns:
        filepath string to data split directory.
    """
    ROOT_URL = 'https://github.com/brendenlake/omniglot/blob/master/python/'
    split_to_name = {'train': 'images_background', 'test': 'images_evaluation'}
    filename = split_to_name[split]
    URL = ROOT_URL + filename + '.zip?raw=true'
    directory = 'paz/datasets/omniglot'
    filepath = get_file(None, URL, cache_subdir=directory, extract=True)
    filepath = os.path.join(os.path.dirname(filepath), filename)
    return filepath


def build_keyname(string):
    """Builds keynames in lower case and without parenthesis.

    # Arguments:
        string: Keyname string.

    # Returns
        String name for easy dictionary access.
    """
    string = os.path.basename(string)
    translations = {ord('('): None, ord(')'): None}
    return string.translate(translations).lower()


def enumerate_filenames(root_path):
    """Enumerates all file names inside given path.

    # Arguments
        root_path: String, path in which to search.

    # Returns
        list of sorted file names inside root path.
    """
    wildcard = os.path.join(root_path, '*')
    directories = glob.glob(wildcard)
    directories = sorted(directories)
    return directories


def load_shot(filepath, shape):
    """Loads images and preprocess it by resizing and normalizing it.

    # Arguments
        filepath: String indicating path to image.
        shape: List of integers indicating new shape (height, width).

    # Returns
        image as numpy array.
    """
    image = load_image(filepath, num_channels=1)
    image = resize_image(image, (shape))
    image = image / 255.0
    return image


def load_shots(shot_filepaths, shape):
    """Loads all images in character directory

    # Arguments:
        shot_filepaths: String. Filepath to character images.
        shape: List of integers indicating new shape (height, width).

    # Returns:
        Image array with all shots
    """
    shots = []
    for shot_filepath in shot_filepaths:
        shots.append(load_shot(shot_filepath, shape))
    return np.array(shots)


def load_characters(character_filepaths, shape):
    """Loads all characters in data directory.

    # Arguments
        character_filepaths: String indicating path to images.
        shape: List of integers indicating new shape (height, width).

    # Returns
        Dictionary with key name character name and value image array.
    """
    characters = {}
    for character_filepath in character_filepaths:
        character_name = build_keyname(character_filepath)
        shot_filepaths = enumerate_filenames(character_filepath)
        shots = load_shots(shot_filepaths, shape)
        characters[character_name] = shots
    return characters


def load(split='train', shape=(28, 28), flat=True):
    """Loads omniglot dataset for in between and within alphabet sampling.

    # Arguments
        split: String. Either `train` or `test`. Indicates which split to load.
        shape: List of two integers indicating resize shape `(H, W)`.
        flat: Boolean. If `True` the returned data dictionary is organized
            using each possible character as a class, with each key being a
            number having as value an image array.
            If `False` the returned data dictionary is organized using as keys
            the language names and as value another dictionary with keys being
            the character number, and as value the image array.
            This is to perform either sampling between alpahabet (`flat=True`)
            or to perform sampling within alphabet (`flat=False`).
            Usually, neural few-shot learning algorithms have been tested using
            in between alphabet sampling, but the original authors tested using
            the more challenging within alphabet sampling.

    # Returns
        dictionary with class names as keys and image numpy arrays as values.
    """
    filepath = download(split)
    language_filepaths = enumerate_filenames(filepath)
    languages = {}
    for language_filepath in language_filepaths:
        language_name = build_keyname(language_filepath)
        character_directories = enumerate_filenames(language_filepath)
        characters = load_characters(character_directories, shape)
        languages[language_name] = characters
    return flatten(languages) if flat else languages


def flatten(dataset):
    """Removes language hierarchy by having classes as each possible character.

    # Arguments:
        dataset: Dictionary with key names language name, and as value a
            dictionary with key names character names, and value an image array

    # Returns:
        Dictionary with key names as numbers and as values image arrays.
    """
    flat_dataset = {}
    flat_key = 0
    for language_name, language in dataset.items():
        for character_name, characters in language.items():
            flat_dataset[flat_key] = characters
            flat_key = flat_key + 1
    return flat_dataset


def sample_between_alphabet(RNG, dataset, num_ways, num_shots, num_tests=1):
    """Samples classification problems with flat dataset.
    Each sample is a meta learning problem with classes from all languages.

    # Arguments:
        RNG: Numpy random number generator.
        dataset: Dictionary.
        num_ways: Int. Number of classes for each meta learning episode.
        num_shots: Int. Number of train images used at each episode.
        num_tests: In. Number of test images at each episode.

    # Returns:
        Two lists. First list has `(train_images, train_labels)` and
        Second list has `(test_images, test_labels)`.
    """
    # dataset is flat, easier for sampling without replacement
    random_classes = RNG.choice(list(dataset.keys()), num_ways, replace=False)
    test_images, test_labels = [], []
    shot_images, shot_labels = [], []
    num_samples = num_shots + num_tests  # num_shots + 1
    for label, class_name in enumerate(random_classes):
        images = RNG.choice(dataset[class_name], num_samples, replace=False)
        labels = np.full(num_samples, label)
        shot_images.append(images[:num_shots])
        shot_labels.append(labels[:num_shots])
        test_images.append(images[num_shots:])
        test_labels.append(labels[num_shots:])
    shot_images = np.concatenate(shot_images, axis=0)
    shot_labels = np.concatenate(shot_labels, axis=0)
    test_images = np.concatenate(test_images, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    return (shot_images, shot_labels), (test_images, test_labels)


def sample_within_alphabet(RNG, dataset, num_ways, num_shots, num_tests=1):
    """Samples classification problems with class hierarchical dataset.
    Each sample is a meta learning problem with classes from the same language.

    # Arguments:
        RNG: Numpy random number generator.
        dataset: Dictionary.
        num_ways: Int. Number of classes for each meta learning episode.
        num_shots: Int. Number of train images used at each episode.
        num_tests: In. Number of test images at each episode.

    # Returns:
        Two lists. First list has `(train_images, train_labels)` and
        Second list has `(test_images, test_labels)`.
    """
    alphabet_name = RNG.choice(list(dataset.keys()))
    alphabet = dataset[alphabet_name]
    reuse = True if num_ways > len(alphabet) else False  # FIX as 2019 Lake
    class_names = RNG.choice(list(alphabet.keys()), num_ways, reuse).tolist()
    test_images, test_labels = [], []
    shot_images, shot_labels = [], []
    num_samples = num_shots + num_tests  # num_shots + 1
    for label, class_name in enumerate(class_names):
        images = RNG.choice(alphabet[class_name], num_samples, replace=False)
        labels = np.full(num_samples, label)
        shot_images.append(images[:num_shots])
        shot_labels.append(labels[:num_shots])
        test_images.append(images[num_shots:])
        test_labels.append(labels[num_shots:])
    shot_images = np.concatenate(shot_images, axis=0)
    shot_labels = np.concatenate(shot_labels, axis=0)
    test_images = np.concatenate(test_images, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    return (shot_images, shot_labels), (test_images, test_labels)


class Generator(Sequence):
    """Data generator for omniglot dataset with meta-learning episodes
    # Arguments
        sampler:
        num_classes: Int. Number of classes for each meta learning episode.
        num_support: Int. Number of train images used at each episode.
        num_queries: In. Number of test images at each episode.
        image_shape: List of integers indicating new shape (height, width).
        num_steps: Int. Number of samples per epoch.
    """
    def __init__(self, sampler, num_classes, num_support, num_queries,
                 image_shape, num_steps=2000):
        self.sampler = sampler
        self.support_shape = (num_classes, num_support, *image_shape)
        self.queries_shape = (num_classes, num_queries, *image_shape)
        self.num_steps = num_steps

    def __len__(self):
        return self.num_steps

    def __getitem__(self, idx):
        (support, support_labels), (queries, queries_labels) = self.sampler()
        support = np.reshape(support, self.support_shape)
        queries = np.reshape(queries, self.queries_shape)
        return {'support': support, 'queries': queries}, queries_labels


def remove_classes(RNG, data, num_classes):
    """Removes classes by randomly taking out keys from data dictionary.

    # Arguments:
        RNG: Numpy random number generator.
        data: Dictionary with keys as class names and values image arrays.

    # Returns:
        Dictionary with number of classes euqal to `num_classes`.
    """
    keys = RNG.choice(len(data.keys()), num_classes, replace=False)
    data = {key: data[key] for key in keys}
    return data


def split_data(data, validation_split):
    """Splits data keys into training and validation.

    # Arguments:
        data: Dictionary with keys as class names and values image arrays.
        validation_split: Float between `[0, 1]`. Porcentange of training
            data to be used for validation.

    # Returns:
        Two dictionaries with train and vlaidation data dictionaries.
    """
    keys = list(data.keys())
    num_train_keys = int(len(keys) * (1 - validation_split))
    train_keys = keys[:num_train_keys]
    valid_keys = keys[num_train_keys:]
    train_data = {key: data[key] for key in train_keys}
    valid_data = {key: data[key] for key in valid_keys}
    return train_data, valid_data


def plot_language(language):
    """Plots all characters in a language

    # Arguments:
        language: Dict with characters names as keys and image arrays as values

    # Returns:
        Image array with all characters.
    """
    characters = []
    for characters_name, images in language.items():
        images = np.expand_dims(images, axis=-1)
        characters.append(make_mosaic(images, (5, 4), 10))
    characters = np.array(characters)
    characters = make_mosaic(characters, (8, 7), 20)
    return characters


@docstring(load)
class Omniglot(Loader):
    def __init__(self, split, shape, flat=True):
        self.shape = shape
        self.flat = flat
        super(Omniglot, self).__init__(None, split, None, 'Omniglot')

    def load_data(self):
        return load(self.split, self.shape, self.flat)
