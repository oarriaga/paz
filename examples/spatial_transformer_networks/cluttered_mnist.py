import numpy as np
import keras


def download():
    URL = (
        "https://github.com/daviddao/spatial-transformer-tensorflow/raw/"
        "7d954f8d57f75c1787b489328c8fb4201ed2fb05/data/"
        "mnist_sequence1_sample_5distortions5x5.npz"
    )
    return keras.utils.get_file(
        "mnist_sequence1_sample_5distortions5x5.npz",
        URL,
        cache_subdir="datasets",
        extract=False,
    )


def to_categorical(y, num_classes=None):
    y = np.array(y, dtype="int").ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical


def process(x, y, shape=(40, 40)):
    x = x.reshape((x.shape[0], *shape, 1))
    return x, y


def load(dataset_path=None, num_classes=10):
    if dataset_path is None:
        dataset_path = download()
    data = np.load(dataset_path)
    x_train, y_train = process(data["X_train"], data["y_train"])
    x_valid, y_valid = process(data["X_valid"], data["y_valid"])
    x_test, y_test = process(data["X_test"], data["y_test"])
    return ((x_train, y_train), (x_valid, y_valid), (x_test, y_test))


if __name__ == "__main__":
    import paz

    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load()
    paz.image.show(
        paz.image.denormalize(paz.draw.mosaic(x_train, border=5, background=1))
    )
