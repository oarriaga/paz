import os
import numpy as np

from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K

from ..backend.image import write_image

from paz.evaluation import evaluateMAP


class DrawInferences(Callback):
    """Saves an image with its corresponding inferences

    # Arguments
        save_path: String. Path in which the images will be saved.
        images: List of numpy arrays of shape.
        pipeline: Function that takes as input an element of ''images''
            and outputs a ''Dict'' with inferences.
        topic: Key to the ''inferences'' dictionary containing as value the
            drawn inferences.
        verbose: Integer. If is bigger than 1 messages would be displayed.
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
        for image_arg, image in enumerate(self.images):
            inferences = self.pipeline(image.copy())
            epoch_name = 'epoch_%03d' % epoch
            save_path = os.path.join(self.save_path, epoch_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            image_name = 'image_%03d.png' % image_arg
            image_name = os.path.join(save_path, image_name)
            write_image(image_name, inferences[self.topic])
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
        verbose: Integer. If is bigger than 1 messages would be displayed.
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


class EvaluateMAP(Callback):
    """Evaluates mean average precision (MAP) of an object detector.

    # Arguments
        data_manager: Data manager and loader class. See ''paz.datasets''
            for examples.
        detector: Tensorflow-Keras model.
        period: Int. Indicates how often the evaluation is performed.
        save_path: Str.
        iou_thresh: Float.
    """
    def __init__(
            self, data_manager, detector, period, save_path, iou_thresh=0.5):
        super(EvaluateMAP, self).__init__()
        self.data_manager = data_manager
        self.detector = detector
        self.period = period
        self.save_path = save_path
        self.dataset = data_manager.load_data()
        self.iou_thresh = iou_thresh
        self.class_names = self.data_manager.class_names
        self.class_dict = {}
        for class_arg, class_name in enumerate(self.class_names):
            self.class_dict[class_name] = class_arg

    def on_epoch_end(self, epoch, logs):
        if (epoch + 1) % self.period == 0:
            result = evaluateMAP(
                self.detector,
                self.dataset,
                self.class_dict,
                iou_thresh=self.iou_thresh,
                use_07_metric=True)

            result_str = 'mAP: {:.4f}\n'.format(result['map'])
            metrics = {'mAP': result['map']}
            for arg, ap in enumerate(result['ap']):
                if arg == 0 or np.isnan(ap):  # skip background
                    continue
                metrics[self.class_names[arg]] = ap
                result_str += '{:<16}: {:.4f}\n'.format(
                    self.class_names[arg], ap)
            print(result_str)

            # Saving the evaluation results
            filename = os.path.join(self.save_path, 'MAP_Evaluation_Log.txt')
            with open(filename, 'a') as eval_log_file:
                eval_log_file.write('Epoch: {}\n{}\n'.format(
                    str(epoch), result_str))
