import numpy as np
import neptune
import os
import sys
import matplotlib.pyplot as plt
import gc
import glob

from pipelines import GeneratedImageGenerator
from paz.abstract import GeneratingSequence

from model import EPOSActivationOutput, epos_loss_wrapped

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

np.set_printoptions(threshold=sys.maxsize)

class NeptuneCallback(Callback):
    def __init__(self, model, save_path, log_interval=100):
        self.model = model
        self.log_interval = log_interval
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs={}):
        for log_name, log_value in logs.items():
            neptune.log_metric(log_name, log_value)

        if epoch%self.log_interval == 0:
            self.model.save(os.path.join(self.save_path, 'epos_{}.h5'.format(epoch)))


class PlotImagesCallback(Callback):
    def __init__(self, model, sequence, num_objects, num_fragments, fragment_centers, object_extent, neptune_logging=False):
        self.model = model
        self.sequence = sequence
        self.neptune_logging = neptune_logging
        self.num_objects = num_objects
        self.num_fragments = num_fragments
        self.fragment_centers = fragment_centers
        self.object_extent = object_extent

    def on_epoch_end(self, epoch_index, logs=None):
        sequence_iterator = self.sequence.__iter__()
        batch = next(sequence_iterator)
        images_original = (batch[0]['input_image'] * 255).astype(np.int)
        
        predicted_epos_output = self.model.predict(batch[0]['input_image'])
        real_epos_output = batch[1]['output']

        # Possible memory leak?
        # https://medium.com/dive-into-ml-ai/dealing-with-memory-leak-issue-in-keras-model-training-e703907a6501
        gc.collect()
        K.clear_session()

        #images_original = np.load("/home/fabian/.keras/tless_obj05/epos/train/image_original/image_original_0000003.npy")
        #images_original = images_original[np.newaxis, ...]
        #real_epos_output_np = np.load("/home/fabian/.keras/tless_obj05/epos/train/epos_output/epos_output_0000000.npy")
        #print("Array diff: {}".format(real_epos_output_np - real_epos_output[0]))
        #np.save("diff.npy", real_epos_output_np - real_epos_output[0])

        #real_epos_output = real_epos_output[np.newaxis, ...]
        #predicted_epos_output = np.load("/home/fabian/.keras/tless_obj05/epos/train/epos_output/epos_output_0000003.npy")
        #predicted_epos_output = predicted_epos_output[np.newaxis, ...]

        num_columns = 7
        num_rows = 3

        fig, ax = plt.subplots(num_rows, num_columns)
        fig.set_size_inches(16, 12)

        col_names = ["Input image", "Real object segmentation", "Real fragments", "Real fragment coordinates", "Predicted object segmentation", "Predicted fragments", "Predicted fragment coordinates"]

        for i in range(num_columns):
            ax[0, i].set_title(col_names[i])
            for j in range(num_rows):
                ax[j, i].get_xaxis().set_visible(False)
                ax[j, i].get_yaxis().set_visible(False)

        for i in range(num_rows):
            print("Epos output shape: {}".format(real_epos_output.shape))
            epos_output_objects = real_epos_output[i, :, :, :self.num_objects+1]
            epos_output_fragments = real_epos_output[i, :, :, self.num_objects+1:self.num_objects+1+self.num_fragments]
            epos_output_fragments_coords = real_epos_output[i, :, :, self.num_objects + 1 + self.num_fragments:]

            ax[i, 0].imshow(images_original[i])
            image_objects = self._object_from_epos_output(epos_output_objects)
            ax[i, 1].imshow(image_objects)
            image_fragments, epos_output_fragments_argmax = self._fragment_from_epos_output(epos_output_fragments, self.fragment_centers, self.object_extent)
            ax[i, 2].imshow(image_fragments)
            image_fragment_coords = self._fragment_coords_from_epos_output(epos_output_fragments_coords, epos_output_fragments_argmax)
            ax[i, 3].imshow(image_fragment_coords)

            epos_output_objects = predicted_epos_output[i, :, :, :self.num_objects+1]
            epos_output_fragments = predicted_epos_output[i, :, :, self.num_objects+1:self.num_objects+1+self.num_fragments]
            epos_output_fragments_coords = predicted_epos_output[i, :, :, self.num_objects + 1 + self.num_fragments:]

            image_objects = self._object_from_epos_output(epos_output_objects)
            ax[i, 4].imshow(image_objects)
            image_fragments, epos_output_fragments_argmax = self._fragment_from_epos_output(epos_output_fragments, self.fragment_centers, self.object_extent)
            ax[i, 5].imshow(image_fragments)
            image_fragment_coords = self._fragment_coords_from_epos_output(epos_output_fragments_coords, epos_output_fragments_argmax)
            ax[i, 6].imshow(image_fragment_coords)

        plt.tight_layout()
        plt.show()
        # plt.savefig(os.path.join(self.save_path, "images/plot-epoch-{}.png".format(epoch_index)))

        if self.neptune_logging:
            neptune.log_image('plot', fig, image_name="epoch_{}.png".format(epoch_index))

        plt.clf()
        plt.close(fig)

    def _object_from_epos_output(self, epos_output_objects):
        colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
        image_objects = np.zeros((epos_output_objects.shape[0], epos_output_objects.shape[1], 3))
        for i in range(epos_output_objects.shape[0]):
            for j in range(epos_output_objects.shape[1]):
                if np.argmax(epos_output_objects[i, j]) == 0:
                    image_objects[i, j] = np.array([0., 0., 0.])
                else:
                    image_objects[i, j] = colors[np.argmax(epos_output_objects[i, j])-1]
        return image_objects.astype("uint8")

    def _fragment_from_epos_output(self, epos_output_fragments, fragment_centers, object_extent, fragment_threshold=0.05):
        # First calculate colors for the fragments
        fragment_center_to_color = list()
        for i, fragment_center in enumerate(fragment_centers):
            fragment_color = fragment_center/(object_extent/2)
            # Clip the values so that the lowest value is 0.2. If it is 0, some
            # fragments are nearly completely black
            fragment_color = np.clip(((fragment_color + 1)/2), 0.2, 1)* 255.
            fragment_color = fragment_color.astype("uint8")
            fragment_center_to_color.append(fragment_color)

        # Turn the one-hot representation into a one where the number indicates
        # which fragment the pixel belongs to
        # If the predicted value for all fragments is low --> 0 (background)
        epos_output_fragments[epos_output_fragments < fragment_threshold] = 0
        epos_output_fragments_argmax = np.argmax(epos_output_fragments, axis=-1)
        image_fragments = np.zeros((epos_output_fragments.shape[0], epos_output_fragments.shape[1], 3))
        for i in range(epos_output_fragments.shape[0]):
            for j in range(epos_output_fragments.shape[1]):
                if (np.sum(epos_output_fragments[i, j]) == 0.):
                    image_fragments[i, j] = np.array([0., 0., 0.])
                    epos_output_fragments_argmax[i, j] = -1.
                else:
                    image_fragments[i, j] = fragment_center_to_color[epos_output_fragments_argmax[i, j]]

        return image_fragments.astype("uint8"), epos_output_fragments_argmax

    def _fragment_coords_from_epos_output(self, epos_output_fragment_coords, epos_output_fragments_argmax):
        image_fragment_coords = np.zeros((epos_output_fragment_coords.shape[0], epos_output_fragment_coords.shape[1], 3))

        for i in range(epos_output_fragment_coords.shape[0]):
            for j in range(epos_output_fragment_coords.shape[1]):
                if epos_output_fragments_argmax[i, j] == -1.:
                    image_fragment_coords[i, j] = np.array([-1, -1, -1])
                else:
                    image_fragment_coords[i, j] = epos_output_fragment_coords[i, j, 3*epos_output_fragments_argmax[i, j]:3*epos_output_fragments_argmax[i, j]+3]

        return ((image_fragment_coords+1)*127.5).astype("uint8")

if __name__ == "__main__":
    fragment_centers = np.load("/home/fabian/Dokumente/epos_data/fragment_centers.npy")

    num_objects = 1
    num_fragments = 64
    num_output_channels = (4 * num_objects * num_fragments + num_objects + 1)

    background_image_paths = glob.glob(os.path.join("/home/fabian/.keras/backgrounds", '*.jpg'))
    processor = GeneratedImageGenerator(os.path.join("/home/fabian/Dokumente/epos", "test"), background_image_paths, 128, num_output_channels)
    sequence = GeneratingSequence(processor, 3, 1)

    epos_loss = epos_loss_wrapped(num_objects=num_objects, num_fragments=num_fragments)
    model = load_model("/home/fabian/Dokumente/epos_4200.h5", custom_objects={"relu6": tf.nn.relu6, "EPOSActivationOutput": EPOSActivationOutput, "epos_loss": epos_loss})

    callback = PlotImagesCallback(model, sequence, num_objects=1, num_fragments=64, fragment_centers=fragment_centers, object_extent=np.array([0.1, 0.15, 0.2]))
    callback.on_epoch_end(0)