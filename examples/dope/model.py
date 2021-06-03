import os

from tensorflow.keras.layers import Conv2D, Layer, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.applications.vgg19 import VGG19

import matplotlib.pyplot as plt
import numpy as np

import neptune


class LogSoftmax(Layer):
    def __init__(self, **kwargs):
        super(LogSoftmax, self).__init__(**kwargs)

    def call(self, inputs):
        print(inputs.shape)
        return inputs


class NeptuneLogger(Callback):

    def __init__(self, model):
        self.model = model

    def on_epoch_end(self, epoch, logs={}):
        for log_name, log_value in logs.items():
            neptune.log_metric(log_name, log_value)

        if epoch%25 == 0:
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

        num_rows = 5
        num_cols = 1
        fig, ax = plt.subplots(num_rows, num_cols)
        cols = ["Input image", "Ground truth", "Predicted image", "Predicted error"]

        for i in range(num_rows):
            #for j in range(num_cols):
            ax[i].get_xaxis().set_visible(False)
            ax[i].get_yaxis().set_visible(False)

        # Show original image in the first row
        ax[0].imshow(original_images[0], vmin=0.0, vmax=1.0)
        #for i in range(1, num_cols):
        ax[0].axis('off')

        # Show real belief and affinity maps of the first stage in the second row
        ax[1].set_title("Real belief maps first stage")
        #for i in range(num_cols):
        ax[1].imshow(batch[1]["belief_maps_stage_1"][0, :, :, 0], cmap='gray')
        #ax[1, 1].imshow(batch[1]["belief_maps_stage_1"][0, :, :, 1], cmap='gray')
        #ax[1, 0].imshow(batch[1]["belief_maps_stage_1"][0, :, :, 2], cmap='gray')
        #ax[1, 1].imshow(batch[1]["belief_maps_stage_1"][0, :, :, 3], cmap='gray')
        #ax[1, 2].imshow(batch[1]["affinity_maps_stage_1"][0, :, :, 0], cmap='gray', vmin=-1.0, vmax=1.0)
        #ax[1, 3].imshow(batch[1]["affinity_maps_stage_1"][0, :, :, 1], cmap='gray', vmin=-1.0, vmax=1.0)

        # Show predicted belief and affinity maps of the first stage in the third row
        ax[2].set_title("Predicted belief maps first stage")
        #for i in range(num_cols):
        ax[2].imshow(predictions[0][0, :, :, 0], cmap='gray', vmin=0.0, vmax=1.0)
        #ax[2, 1].imshow(predictions[0][0, :, :, 1], cmap='gray', vmin=0.0, vmax=1.0)
        #ax[2, 0].imshow(predictions[0][0, :, :, 2], cmap='gray', vmin=0.0, vmax=1.0)
        #ax[2, 1].imshow(predictions[0][0, :, :, 3], cmap='gray', vmin=0.0, vmax=1.0)
        #ax[2, 2].imshow(predictions[1][0, :, :, 0], cmap='gray', vmin=-1.0, vmax=1.0)
        #ax[2, 3].imshow(predictions[1][0, :, :, 1], cmap='gray', vmin=-1.0, vmax=1.0)

        # Show real belief and affinity maps of the last stage in the third row
        ax[3].set_title("Real belief maps last stage")
        #for i in range(num_cols):
        ax[3].imshow(batch[1]["belief_maps_stage_{}".format(self.num_stages)][0, :, :, 0], cmap='gray')
        #ax[3, 1].imshow(batch[1]["belief_maps_stage_{}".format(self.num_stages)][0, :, :, 1], cmap='gray')
        #ax[3, 0].imshow(batch[1]["belief_maps_stage_{}".format(self.num_stages)][0, :, :, 2], cmap='gray')
        #ax[3, 1].imshow(batch[1]["belief_maps_stage_{}".format(self.num_stages)][0, :, :, 3], cmap='gray')
        #ax[3, 2].imshow(batch[1]["affinity_maps_stage_{}".format(self.num_stages)][0, :, :, 0], cmap='gray', vmin=-1.0, vmax=1.0)
        #ax[3, 3].imshow(batch[1]["affinity_maps_stage_{}".format(self.num_stages)][0, :, :, 1], cmap='gray', vmin=-1.0, vmax=1.0)

        # Show predicted belief and affinity maps of the last stage in the fourth row
        ax[4].set_title("Predicted belief maps last stage")
        #for i in range(num_cols):
        ax[4].imshow(predictions[-1][0, :, :, 0], cmap='gray', vmin=0.0, vmax=1.0)
        #ax[4, 1].imshow(predictions[-2][0, :, :, 1], cmap='gray', vmin=0.0, vmax=1.0)
        #ax[4, 0].imshow(predictions[-2][0, :, :, 2], cmap='gray', vmin=0.0, vmax=1.0)
        #ax[4, 1].imshow(predictions[-2][0, :, :, 3], cmap='gray', vmin=0.0, vmax=1.0)
        #ax[4, 2].imshow(predictions[-1][0, :, :, 0], cmap='gray', vmin=-1.0, vmax=1.0)
        #ax[4, 3].imshow(predictions[-1][0, :, :, 1], cmap='gray', vmin=-1.0, vmax=1.0)

        plt.tight_layout()

        plt.savefig("plot-epoch-{}.png".format(epoch_index))

        if self.neptune_logging:
            neptune.log_image('plot', fig, image_name="epoch_{}.png".format(epoch_index))

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
    output = Conv2D(out_channels, kernel_size=(1, 1), padding='same', activation="linear", name="{}_stage_{}".format(name, num_stage))(x)

    return output


def DOPE(num_classes=2, num_belief_maps=9, num_affinity_maps=16, image_shape=(400, 400, 3), num_stages=1):
    # VGG-19 backend
    model_vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=image_shape)
    model_vgg19.trainable = True

    output_vgg19 = model_vgg19.get_layer('block4_conv3').output

    # Add some convolutional layers on top
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(output_vgg19)
    conv_out = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)

    # Create the first stage
    belief_map_stage01 = create_stage(conv_out, num_belief_maps, num_stage=1, name="belief_maps", activation_last_layer='tanh')
    affinity_map_stage01 = create_stage(conv_out, num_affinity_maps, num_stage=1, name="affinity_maps", activation_last_layer='sigmoid')

    if num_stages == 1:
        model = Model(inputs=[model_vgg19.get_layer("input_1").input], outputs=[belief_map_stage01])
        return model

    # Create the second stage
    concatenate_stage02 = Concatenate()([conv_out, belief_map_stage01])
    belief_map_stage02 = create_stage(concatenate_stage02, num_belief_maps, num_stage=2, name="belief_maps", activation_last_layer='tanh')
    affinity_map_stage02 = create_stage(concatenate_stage02, num_affinity_maps, num_stage=2, name="affinity_maps", activation_last_layer='sigmoid')

    if num_stages == 2:
        model = Model(inputs=[model_vgg19.get_layer("input_1").input], outputs=[belief_map_stage01,
                                                                               belief_map_stage02])
        print(model.summary())
        return model

    # Create the third stage
    concatenate_stage03 = Concatenate()([conv_out, belief_map_stage02])
    belief_map_stage03 = create_stage(concatenate_stage03, num_belief_maps, num_stage=3, name="belief_maps", activation_last_layer='tanh')
    affinity_map_stage03 = create_stage(concatenate_stage03, num_affinity_maps, num_stage=3, name="affinity_maps", activation_last_layer='sigmoid')

    if num_stages == 3:
        model = Model(inputs=[model_vgg19.get_layer("input_1").input], outputs=[belief_map_stage01,
                                                                               belief_map_stage02,
                                                                               belief_map_stage03])
        print(model.summary())
        return model

    # Create the fourth stage
    concatenate_stage04 = Concatenate()([conv_out, belief_map_stage03, affinity_map_stage03])
    belief_map_stage04 = create_stage(concatenate_stage04, num_belief_maps, num_stage=4, name="belief_maps", activation_last_layer='tanh')
    affinity_map_stage04 = create_stage(concatenate_stage04, num_affinity_maps, num_stage=4, name="affinity_maps", activation_last_layer='sigmoid')

    if num_stages == 4:
        model = Model(inputs=[model_vgg19.get_layer("input_1").input], outputs=[belief_map_stage01, affinity_map_stage01,
                                                                               belief_map_stage02, affinity_map_stage02,
                                                                               belief_map_stage03, affinity_map_stage03,
                                                                               belief_map_stage04, affinity_map_stage04])

        print(model.summary())
        return model

    # Create the fifth stage
    concatenate_stage05 = Concatenate()([conv_out, belief_map_stage04, affinity_map_stage04])
    belief_map_stage05 = create_stage(concatenate_stage05, num_belief_maps, num_stage=5, name="belief_maps", activation_last_layer='tanh')
    affinity_map_stage05 = create_stage(concatenate_stage05, num_affinity_maps, num_stage=5, name="affinity_maps", activation_last_layer='sigmoid')

    if num_stages == 5:
        model = Model(inputs=[model_vgg19.get_layer("input_1").input], outputs=[belief_map_stage01, affinity_map_stage01,
                                                                               belief_map_stage02, affinity_map_stage02,
                                                                               belief_map_stage03, affinity_map_stage03,
                                                                               belief_map_stage04, affinity_map_stage04,
                                                                               belief_map_stage05, affinity_map_stage05])
        print(model.summary())
        return model

    # Create the sixth stage
    concatenate_stage06 = Concatenate()([conv_out, belief_map_stage05, affinity_map_stage05])
    belief_map_stage06 = create_stage(concatenate_stage06, num_belief_maps, num_stage=6, name="belief_maps", activation_last_layer='tanh')
    affinity_map_stage06 = create_stage(concatenate_stage06, num_affinity_maps, num_stage=6, name="affinity_maps", activation_last_layer='sigmoid')

    if num_stages == 6:
        model = Model(inputs=[model_vgg19.get_layer("input_1").input], outputs=[belief_map_stage01, affinity_map_stage01,
                                                                               belief_map_stage02, affinity_map_stage02,
                                                                               belief_map_stage03, affinity_map_stage03,
                                                                               belief_map_stage04, affinity_map_stage04,
                                                                               belief_map_stage05, affinity_map_stage05,
                                                                               belief_map_stage06, affinity_map_stage06])
        print(model.summary())
        return model


if __name__ == "__main__":
    dope = DOPE(num_stages=6)