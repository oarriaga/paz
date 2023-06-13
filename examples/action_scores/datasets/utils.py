import numpy as np


def get_class_names(dataset_name='MNIST'):
    if dataset_name == 'MNIST':
        return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    if dataset_name == 'FashionMNIST':
        return ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    if dataset_name == 'CIFAR10':
        return ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']
    if dataset_name == 'KuzushijiMNIST':
        return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    if dataset_name == 'FER':
        return ['angry', 'disgust', 'fear', 'happy',
                'sad', 'surprise', 'neutral']
    else:
        raise ValueError('Invalid dataset name')


def poison_labels(labels, num_classes, percent=0.1):
    num_samples = len(labels)
    num_poisons = int(num_samples * percent)
    selected_args = np.random.choice(num_samples, num_poisons, replace=False)
    poisoned_labels = labels.copy()
    for arg in selected_args:
        poisoned_labels[arg] = poison_label(poisoned_labels[arg], num_classes)
    return poisoned_labels


def poison_label(label, num_classes):
    valid_class_args = list(range(num_classes))
    valid_class_args.remove(label)
    poison = np.random.choice(valid_class_args, 1)[0]
    return poison


def test_compute_poison(num_classes=10, num_samples=100):
    labels = np.random.randint(0, num_classes, num_samples)
    for label_arg, label in enumerate(labels):
        poison = poison_label(label, num_classes)
        assert poison != label


def test_num_poisoned_labels(num_classes=10, num_samples=100, percent=0.1):
    labels = np.random.randint(0, num_classes, num_samples)
    poisoned_labels = poison_labels(labels, num_classes, percent)
    num_poisons = np.sum(labels != poisoned_labels)
    assert num_poisons == int(percent * num_samples)


test_compute_poison()
test_num_poisoned_labels()
