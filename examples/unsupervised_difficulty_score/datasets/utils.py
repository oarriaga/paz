def get_class_names(self, dataset_name='MNIST'):
    if dataset_name == 'MNIST':
        return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    if dataset_name == 'CIFAR10':
        return ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']
    else:
        raise ValueError('Invalid dataset name')
