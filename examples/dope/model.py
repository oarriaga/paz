import os

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
    return tf.reduce_mean(squared_difference, axis=[1, 2])


class LogSoftmax(Layer):
    def __init__(self, **kwargs):
        super(LogSoftmax, self).__init__(**kwargs)

    def call(self, inputs):
        print(inputs.shape)
        return inputs


class NeptuneLogger(Callback):

    def __init__(self, model, log_interval):
        self.model = model
        self.log_interval = log_interval

    def on_epoch_end(self, epoch, logs={}):
        for log_name, log_value in logs.items():
            neptune.log_metric(log_name, log_value)

        if epoch%self.log_interval == 0:
            self.model.save('dope_model_epoch_{}.pkl'.format(epoch))
            neptune.log_artifact('dope_model_epoch_{}.pkl'.format(epoch))


class PlotImagesCallback(Callback):
    def __init__(self, model, sequence, num_stages, neptune_logging=False):
        self.model = model
        self.sequence = sequence
        self.neptune_logging = neptune_logging
        self.num_stages = num_stages

    def on_epoch_end(self, epoch_index, logs=None):
        batch = self.sequence.__getitem__(0)
        predictions = self.model.predict(batch[0]['input_1'])

        original_images = batch[0]['input_1']

        # num_stages + one row for input image and one row for ground truth
        num_rows = self.num_stages + 2
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
            ax[1, i].imshow(batch[1]["belief_maps_stage_1"][0, :, :, i], cmap='gray', vmin=0.0, vmax=1.0)

        # Show the predicted belief maps
        for i in range(self.num_stages):
            ax[2 + i, 4].set_title("Predicted belief maps stage number {}".format(i))
            for col_number in range(num_cols):
                ax[2 + i, col_number].imshow(predictions[i][0, :, :, col_number], cmap='gray', vmin=0.0, vmax=1.0)

        #plt.tight_layout()
        fig.subplots_adjust(hspace=0.5)

        if self.neptune_logging:
            neptune.log_image('plot', fig, image_name="epoch_{}.png".format(epoch_index))
        else:
            plt.savefig("plot-epoch-{}.png".format(epoch_index))

        plt.clf()
        plt.close(fig)


def create_stage(input, out_channels, num_stage, name, activation_last_layer='sigmoid'):
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
    output = Conv2D(out_channels, kernel_size=(1, 1), padding='same', activation=activation_last_layer, name="{}_stage_{}".format(name, num_stage))(x)

    return output


def DOPE(num_classes=2, num_belief_maps=9, num_affinity_maps=16, image_shape=(400, 400, 3), num_stages=1):
    # VGG-19 backend
    #model_vgg19 = VGG19(weights="imagenet", include_top=False, input_shape=image_shape)
    #model_vgg19.trainable = True
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

    #output_vgg19 = model_vgg19.get_layer('block4_conv3').output

    # Add some convolutional layers on top
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(output_vgg19)
    conv_out = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)

    #conv_out_alternative = Conv2D(num_belief_maps, kernel_size=(1, 1), padding='same', activation="linear",
    #                              name="{}_stage_{}".format("belief_maps", 1))(conv_out)

    # Create the first stage
    belief_map_stage01 = create_stage(conv_out, num_belief_maps, num_stage=1, name="belief_maps", activation_last_layer='linear')
    affinity_map_stage01 = create_stage(conv_out, num_affinity_maps, num_stage=1, name="affinity_maps", activation_last_layer='sigmoid')

    if num_stages == 1:
        model = Model(inputs=[inp], outputs=[belief_map_stage01])
        return model

    # Create the second stage
    concatenate_stage02 = Concatenate()([conv_out, belief_map_stage01])
    belief_map_stage02 = create_stage(concatenate_stage02, num_belief_maps, num_stage=2, name="belief_maps", activation_last_layer='linear')
    affinity_map_stage02 = create_stage(concatenate_stage02, num_affinity_maps, num_stage=2, name="affinity_maps", activation_last_layer='sigmoid')

    if num_stages == 2:
        model = Model(inputs=[inp], outputs=[belief_map_stage01, belief_map_stage02])
        print(model.summary())
        return model

    # Create the third stage
    concatenate_stage03 = Concatenate()([conv_out, belief_map_stage02])
    belief_map_stage03 = create_stage(concatenate_stage03, num_belief_maps, num_stage=3, name="belief_maps", activation_last_layer='linear')
    affinity_map_stage03 = create_stage(concatenate_stage03, num_affinity_maps, num_stage=3, name="affinity_maps", activation_last_layer='sigmoid')

    if num_stages == 3:
        model = Model(inputs=[inp], outputs=[belief_map_stage01, belief_map_stage02, belief_map_stage03])
        print(model.summary())
        return model

    # Create the fourth stage
    concatenate_stage04 = Concatenate()([conv_out, belief_map_stage03])
    belief_map_stage04 = create_stage(concatenate_stage04, num_belief_maps, num_stage=4, name="belief_maps", activation_last_layer='linear')
    affinity_map_stage04 = create_stage(concatenate_stage04, num_affinity_maps, num_stage=4, name="affinity_maps", activation_last_layer='sigmoid')

    if num_stages == 4:
        model = Model(inputs=[inp], outputs=[belief_map_stage01, belief_map_stage02, belief_map_stage03, belief_map_stage04])

        print(model.summary())
        return model

    # Create the fifth stage
    concatenate_stage05 = Concatenate()([conv_out, belief_map_stage04])
    belief_map_stage05 = create_stage(concatenate_stage05, num_belief_maps, num_stage=5, name="belief_maps", activation_last_layer='linear')
    affinity_map_stage05 = create_stage(concatenate_stage05, num_affinity_maps, num_stage=5, name="affinity_maps", activation_last_layer='sigmoid')

    if num_stages == 5:
        model = Model(inputs=[inp], outputs=[belief_map_stage01, belief_map_stage02, belief_map_stage03, belief_map_stage04, belief_map_stage05])
        print(model.summary())
        return model

    # Create the sixth stage
    concatenate_stage06 = Concatenate()([conv_out, belief_map_stage05])
    belief_map_stage06 = create_stage(concatenate_stage06, num_belief_maps, num_stage=6, name="belief_maps", activation_last_layer='linear')
    affinity_map_stage06 = create_stage(concatenate_stage06, num_affinity_maps, num_stage=6, name="affinity_maps", activation_last_layer='sigmoid')

    if num_stages == 6:
        model = Model(inputs=[inp], outputs=[belief_map_stage01, belief_map_stage02, belief_map_stage03, belief_map_stage04, belief_map_stage05, belief_map_stage06])
        print(model.summary())
        return model


if __name__ == "__main__":
    dope = DOPE(num_stages=6)