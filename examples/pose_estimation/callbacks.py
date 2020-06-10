from tensorflow.keras.callbacks import Callback
import numpy as np


class ChangeGenerator(Callback):
    """Change scene when a monitored quantity has stopped improving.

    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing
    """

    def __init__(self, generator, monitor, patience=0,
                 min_delta=0, verbose=0, mode='min'):
        super(ChangeGenerator, self).__init__()

        if mode not in ['min', 'max']:
            raise ValueError('Invalid mode', mode)

        self.generator = generator
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta = self.min_delta * -1
        else:
            self.monitor_op = np.greater
            self.min_delta = self.min_delta * 1
        self.wait = 0
        self.stopped_epoch = 0

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used (resets internal variables)
        self.wait, self.stopped_epoch = 0, 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait = self.wait + 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            raise ValueError('Unavailable metric:', self.monitor)
        return monitor_value

    # def change_scene(self):
        # self.generator.scene.translate = translate
        # self.generator.scene.
