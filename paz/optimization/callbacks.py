import os
import numpy as np

from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K

from ..core import ops


class DrawInferences(Callback):
    """Saves an image with its corresponding inferences

    # Arguments
        save_path: String. Path in which the images will be saved.
        sequencer: Sequencer with __getitem__ function for calling a batch.
        inferencer: Paz Processor for performing inference.
        verbose: Integer. If is bigger than 1 a message with the learning
            rate decay will be displayed during optimization.
    """
    def __init__(self, save_path, images, pipeline, topic='image', verbose=1):
        super(DrawInferences, self).__init__()
        self.save_path = os.path.join(save_path, 'images')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.pipeline = pipeline
        self.images = images
        self.topic = topic
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        for image_arg, image in enumerate(self.images.copy()):
            inferences = self.pipeline({self.topic: image})
            epoch_name = 'epoch_%03d' % epoch
            save_path = os.path.join(self.save_path, epoch_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            image_name = 'image_%03d.png' % image_arg
            image_name = os.path.join(save_path, image_name)
            ops.save_image(image_name, inferences[self.topic])
        if self.verbose:
            print('Saving predicted images in:', self.save_path)


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
