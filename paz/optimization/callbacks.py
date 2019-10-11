from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import numpy as np


class LearningRateScheduler(Callback):
    """ Callback for reducing learning rate at specific epochs.

    # Arguments
        learning_rate: float. Indicates the starting learning rate.
        gamma_decay: float. In an scheduled epoch the learning rate
            is multiplied by this factor.
        scheduled_epochs: List of integers. Indicates in which epochs
            the learning rate will be multiplied by the gamma decay factor.
        verbose: Integer. If is bigger than 1 a message with the learning
            rate decay will be displayed during optimization.
    """
    def __init__(
            self, learning_rate, gamma_decay, scheduled_epochs, verbose=1):
        super(LearningRateScheduler, self).__init__()
        self.learning_rate = learning_rate
        self.gamma_decay = gamma_decay
        self.scheduled_epochs = scheduled_epochs
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        learning_rate = float(K.get_value(self.model.optimizer.lr))
        learning_rate = self.schedule(epoch)
        if not isinstance(learning_rate, (float, np.float32, np.float64)):
            raise ValueError('Learning rate should be float.')
        K.set_value(self.model.optimizer.lr, learning_rate)
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler reducing learning '
                  'rate to %s.' % (epoch + 1, learning_rate))

    def schedule(self, epoch):
        if epoch in self.scheduled_epochs:
            self.learning_rate = self.learning_rate * self.gamma_decay
        return self.learning_rate
