import numpy as np


class ClutteredMNIST(object):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def to_categorical(self, y, num_classes=None):
        y = np.array(y, dtype='int').ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, num_classes))
        categorical[np.arange(n), y] = 1
        return categorical

    def load(self):
        num_classes = 10
        data = np.load(self.dataset_path)
        x_train = data['x_train']
        x_train = x_train.reshape((x_train.shape[0], 60, 60, 1))
        y_train = np.argmax(data['y_train'], axis=-1)
        y_train = self.to_categorical(y_train, num_classes)
        train_data = (x_train, y_train)

        x_val = data['x_valid']
        x_val = x_val.reshape((x_val.shape[0], 60, 60, 1))
        y_val = np.argmax(data['y_valid'], axis=-1)
        y_val = self.to_categorical(y_val, num_classes)
        val_data = (x_val, y_val)

        x_test = data['x_test']
        x_test = x_test.reshape((x_test.shape[0], 60, 60, 1))
        y_test = np.argmax(data['y_test'], axis=-1)
        y_test = self.to_categorical(y_test, num_classes)
        test_data = (x_test, y_test)
        return(train_data, val_data, test_data)
