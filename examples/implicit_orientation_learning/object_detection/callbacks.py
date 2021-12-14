import numpy as np
import random
import os
import matplotlib.pyplot as plt
import neptune
from tensorflow.keras.callbacks import Callback

from paz.pipelines.detection import DetectSingleShot
from paz.processors import SequentialProcessor
from paz.pipelines import RandomizeRenderedImage
import paz.processors as pr


class PlotImagesCallback(Callback):

    def __init__(self, model, data, background_image_paths, neptune_logging=False):
        self.model = model
        self.data = data
        self.neptune_logging = neptune_logging
        self.background_image_paths = background_image_paths

    def on_epoch_end(self, epoch, logs=None):
        num_predictions = 4
        pipeline = SequentialProcessor([pr.CastImage(np.single),
                                        DetectSingleShot(self.model, ["Background", "TLESS 14"], 0.6, 0.45),
                                        pr.UnpackDictionary(['image', 'boxes2D']),
                                        pr.ControlMap(pr.CastImage(np.int), [0], [0])])
        augment_background = RandomizeRenderedImage(image_paths=self.background_image_paths, num_occlusions=0)

        fig, ax = plt.subplots(1, num_predictions)
        fig.set_size_inches(10, 6)

        for i in range(num_predictions):
            idx = random.randrange(len(self.data))
            image = self.data[idx]['image']
            alpha_mask = self.data[idx]['alpha_mask']

            image = augment_background(image, alpha_mask)
            prediction = pipeline(image)

            ax[i].imshow(prediction[0])

        plt.show()

        if self.neptune_logging:
            neptune.log_image('plot', fig, image_name="epoch_{}.png".format(epoch))

        plt.clf()
        plt.close(fig)


class NeptuneLogger(Callback):

    def __init__(self, model, log_interval, save_path):
        self.model = model
        self.log_interval = log_interval
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs={}):
        for log_name, log_value in logs.items():
            neptune.log_metric(log_name, log_value)

        if epoch%self.log_interval == 0:
            self.model.save_weights(os.path.join(self.save_path, 'ssd_300_detection_{}_weights.h5'.format(epoch)))
