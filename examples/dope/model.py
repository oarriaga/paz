import os
from itertools import chain

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Layer, Concatenate, Input, MaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.applications.vgg19 import VGG19

import matplotlib.pyplot as plt
import numpy as np

import neptune


def custom_mse(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=[1, 2, 3])


class LogSoftmax(Layer):
    def __init__(self, **kwargs):
        super(LogSoftmax, self).__init__(**kwargs)

    def call(self, inputs):
        print(inputs.shape)
        return inputs


class NeptuneLogger(Callback):

    def __init__(self, model, log_interval, save_path):
        self.model = model
        self.log_interval = log_interval
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs={}):
        for log_name, log_value in logs.items():
            neptune.log_metric(log_name, log_value)

        if epoch%self.log_interval == 0:
            self.model.save(os.path.join(self.save_path, 'dope_model_epoch_{}.pkl'.format(epoch)))
            #neptune.log_artifact('dope_model_epoch_{}.pkl'.format(epoch))


class PlotImagesCallback(Callback):
    def __init__(self, model, sequence, num_stages, neptune_logging=False, multipleHypotheses=None):
        self.model = model
        self.sequence = sequence
        self.neptune_logging = neptune_logging
        self.num_stages = num_stages
        self.multipleHypotheses = multipleHypotheses

    def on_epoch_end(self, epoch_index, logs=None):
        sequence_iterator = self.sequence.__iter__()
        batch = next(sequence_iterator)

        predictions = self.model.predict(batch[0]['input_1'])

        original_images = batch[0]['input_1']

        # num of hypotheses + one row for input image and one row for ground truth
        num_rows = self.multipleHypotheses.M + 2
        num_cols = 9
        fig, ax = plt.subplots(num_rows, num_cols)
        fig.set_size_inches(18.5, 10.5)

        for i in range(num_rows):
            for j in range(num_cols):
                ax[i, j].get_xaxis().set_visible(False)
                ax[i, j].get_yaxis().set_visible(False)

        # Show original image in the first row
        ax[0, 0].set_title("Input image")
        ax[0, 0].imshow(original_images[0])
        for i in range(1, num_cols):
            ax[0, i].axis('off')

        # Show real belief maps
        ax[1, 4].set_title("Real belief maps")
        for i in range(num_cols):
            ax[1, i].imshow(batch[1]["belief_maps_stage_1_0"][0, :, :, i], cmap='gray', vmin=0.0, vmax=1.0)

        # Show the predicted belief maps
        for i in range(self.multipleHypotheses.M):
            ax[2 + i, 4].set_title("Predicted belief maps last stage, hypotheses #{}".format(i + 1))
            for col_number in range(num_cols):
                ax[2 + i, col_number].imshow(predictions['belief_maps_stage_6_{}'.format(i)][0, :, :, col_number], cmap='gray', vmin=0.0, vmax=1.0)

        plt.tight_layout()
        plt.show()
        fig.subplots_adjust(hspace=0.5)

        if self.neptune_logging:
            neptune.log_image('plot', fig, image_name="epoch_{}.png".format(epoch_index))
        else:
            plt.savefig("plot-epoch-{}.png".format(epoch_index))

        plt.clf()
        plt.close(fig)


def create_stage(input, out_channels, num_stage, name, activation_last_layer='sigmoid', multipleHypotheses=None):
    mid_channels = 128

    # Parameters depend on whether this is the first or an intermediate layer
    if num_stage == 1:
        kernel_size = 3
        num_mid_layers = 2
        final_channels = 512
    else:
        kernel_size = 7
        num_mid_layers = 5
        final_channels = mid_channels

    # Add first convolutional layer
    x = Conv2D(mid_channels, kernel_size=(kernel_size, kernel_size), padding='same', activation='relu')(input)

    # Add intermediate layers
    for _ in range(num_mid_layers):
        x = Conv2D(mid_channels, kernel_size=(kernel_size, kernel_size), padding='same', activation='relu')(x)

    # Second-to-last layer
    x = Conv2D(final_channels, kernel_size=(1, 1), padding='same', activation='relu')(x)

    # Last layer
    if multipleHypotheses is None:
        output = Conv2D(out_channels, kernel_size=(1, 1), padding='same', activation=activation_last_layer, name="{}_stage_{}".format(name, num_stage))(x)
        return output
    else:
        outputs, output_names = multipleHypotheses.multiple_hypotheses_output(x, {"{}_stage_{}".format(name, num_stage): Conv2D(out_channels, kernel_size=(1, 1), padding='same', activation=activation_last_layer)})

        output_dict = dict()
        for output, output_name in zip(outputs, output_names):
            output_dict[output_name] = output

        return output_dict


def DOPE(num_classes=2, num_belief_maps=9, num_affinity_maps=16, image_shape=(400, 400, 3), num_stages=1, multipleHypotheses=None):
    # VGG-19 backend
    inp = Input(shape=image_shape, name='input_1')
    x = Conv2D(64, input_shape=image_shape, kernel_size=(3, 3), padding='same', activation='relu')(inp)
    x = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPool2D()(x)
    x = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPool2D()(x)
    x = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPool2D()(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu')(x)

    output_vgg19 = Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu')(x)

    # Add some convolutional layers on top
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(output_vgg19)
    conv_out = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)

    # Create the first stage
    belief_map_stage01 = create_stage(conv_out, num_belief_maps, num_stage=1, name="belief_maps", activation_last_layer='linear', multipleHypotheses=multipleHypotheses)
    affinity_map_stage01 = create_stage(conv_out, num_affinity_maps, num_stage=1, name="affinity_maps", activation_last_layer='sigmoid')

    if num_stages == 1:
        outputs = {**belief_map_stage01}
        model = Model(inputs=[inp], outputs=outputs)
        return model

    # Create the second stage
    if multipleHypotheses is None:
        concatenate_stage02 = Concatenate()([conv_out, belief_map_stage01])
    else:
        concatenate_stage02 = Concatenate()([conv_out] + list(belief_map_stage01.values()))
    belief_map_stage02 = create_stage(concatenate_stage02, num_belief_maps, num_stage=2, name="belief_maps", activation_last_layer='linear', multipleHypotheses=multipleHypotheses)
    affinity_map_stage02 = create_stage(concatenate_stage02, num_affinity_maps, num_stage=2, name="affinity_maps", activation_last_layer='sigmoid')

    if num_stages == 2:
        outputs = {**belief_map_stage01, **belief_map_stage02}
        model = Model(inputs=[inp], outputs=outputs)
        print(model.summary())
        return model

    # Create the third stage
    if multipleHypotheses is None:
        concatenate_stage03 = Concatenate()([conv_out, belief_map_stage02])
    else:
        concatenate_stage03 = Concatenate()([conv_out] + list(belief_map_stage02.values()))
    belief_map_stage03 = create_stage(concatenate_stage03, num_belief_maps, num_stage=3, name="belief_maps", activation_last_layer='linear', multipleHypotheses=multipleHypotheses)
    affinity_map_stage03 = create_stage(concatenate_stage03, num_affinity_maps, num_stage=3, name="affinity_maps", activation_last_layer='sigmoid')

    if num_stages == 3:
        outputs = {**belief_map_stage01, **belief_map_stage02, **belief_map_stage03}
        model = Model(inputs=[inp], outputs=outputs)
        print(model.summary())
        return model

    # Create the fourth stage
    if multipleHypotheses is None:
        concatenate_stage04 = Concatenate()([conv_out, belief_map_stage03])
    else:
        concatenate_stage04 = Concatenate()([conv_out] + list(belief_map_stage03.values()))
    belief_map_stage04 = create_stage(concatenate_stage04, num_belief_maps, num_stage=4, name="belief_maps", activation_last_layer='linear', multipleHypotheses=multipleHypotheses)
    affinity_map_stage04 = create_stage(concatenate_stage04, num_affinity_maps, num_stage=4, name="affinity_maps", activation_last_layer='sigmoid')

    if num_stages == 4:
        outputs = {**belief_map_stage01, **belief_map_stage02, **belief_map_stage03, **belief_map_stage04}
        model = Model(inputs=[inp], outputs=outputs)

        print(model.summary())
        return model

    # Create the fifth stage
    if multipleHypotheses is None:
        concatenate_stage05 = Concatenate()([conv_out, belief_map_stage04])
    else:
        concatenate_stage05 = Concatenate()([conv_out] + list(belief_map_stage04.values()))
    belief_map_stage05 = create_stage(concatenate_stage05, num_belief_maps, num_stage=5, name="belief_maps", activation_last_layer='linear', multipleHypotheses=multipleHypotheses)
    affinity_map_stage05 = create_stage(concatenate_stage05, num_affinity_maps, num_stage=5, name="affinity_maps", activation_last_layer='sigmoid')

    if num_stages == 5:
        outputs = {**belief_map_stage01, **belief_map_stage02, **belief_map_stage03, **belief_map_stage04, **belief_map_stage05}
        model = Model(inputs=[inp], outputs=outputs)
        print(model.summary())
        return model

    # Create the sixth stage
    if multipleHypotheses is None:
        concatenate_stage06 = Concatenate()([conv_out, belief_map_stage05])
    else:
        concatenate_stage06 = Concatenate()([conv_out] + list(belief_map_stage05.values()))
    belief_map_stage06 = create_stage(concatenate_stage06, num_belief_maps, num_stage=6, name="belief_maps", activation_last_layer='linear', multipleHypotheses=multipleHypotheses)
    affinity_map_stage06 = create_stage(concatenate_stage06, num_affinity_maps, num_stage=6, name="affinity_maps", activation_last_layer='sigmoid')

    if num_stages == 6:
        outputs = {**belief_map_stage01, **belief_map_stage02, **belief_map_stage03, **belief_map_stage04, **belief_map_stage05, **belief_map_stage06}
        model = Model(inputs=[inp], outputs=outputs)
        print(model.summary())
        plot_model(model, "model.png")
        return model


if __name__ == "__main__":
    dope = DOPE(num_stages=6)