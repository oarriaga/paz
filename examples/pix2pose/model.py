import numpy as np
import sys
import os
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Conv2D, Activation, UpSampling2D, Dense, Conv2DTranspose
from tensorflow.keras.layers import Dropout, Input, Flatten, Reshape, LeakyReLU, BatchNormalization, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import tensorflow as tf


def transformer_loss(real_depth_image, predicted_depth_image):
    print("shape of real_depth_image: {}".format(real_depth_image.numpy().shape))
    print("shape of predicted_depth_image: {}".format(predicted_depth_image.numpy().shape))

    plt.imshow(predicted_depth_image.numpy()[0])
    plt.show()

    plt.imshow(predicted_depth_image.numpy()[1])
    plt.show()
    #print(predicted_depth_image.numpy()[0])
    return K.mean(K.square(predicted_depth_image - real_depth_image), axis=-1)


def loss_color(real_color_image, predicted_color_image):
    # Calculate masks for the object and the background
    mask_object = tf.repeat(tf.expand_dims(tf.math.reduce_max(tf.math.ceil(real_color_image), axis=-1), axis=-1), repeats=3, axis=-1)
    mask_background = tf.ones(tf.shape(mask_object)) - mask_object

    # Get the number of pixels
    num_pixels = tf.math.reduce_prod(tf.shape(real_color_image)[1:3])
    beta = 3

    # Calculate the difference between the real and predicted images including the mask
    diff_object = tf.math.abs(predicted_color_image*mask_object - real_color_image*mask_object)
    diff_background = tf.math.abs(predicted_color_image*mask_background - real_color_image*mask_background)

    # Calculate the total loss
    loss_colors = tf.cast((1/num_pixels), dtype=tf.float32)*(beta*tf.math.reduce_sum(diff_object, axis=[1, 2, 3]) + tf.math.reduce_sum(diff_background, axis=[1, 2, 3]))
    #print(loss_colors)

    return loss_colors


def loss_error(real_error_image, predicted_error_image):
    # Get the number of pixels
    num_pixels = tf.math.reduce_prod(tf.shape(real_error_image)[1:3])
    loss_error = tf.cast((1/num_pixels), dtype=tf.float32)*(tf.math.reduce_sum(tf.math.square(predicted_error_image - tf.clip_by_value(tf.math.abs(real_error_image), tf.float32.min, 1.)), axis=[1, 2, 3]))
    #print(loss_error)

    return loss_error


class PlotImagesCallback(Callback):
    def __init__(self, model, sequence, save_path):
        self.save_path = save_path
        self.model = model
        self.sequence = sequence

    def on_epoch_end(self, epoch_index, logs=None):
        batch = self.sequence.__getitem__(0)
        predictions = self.model.predict(batch[0]['input_image'])

        original_images = (batch[0]['input_image'] * 255).astype(np.int)
        color_images = ((batch[1]['color_output'] + 1) * 127.5).astype(np.int)
        predictions[0] = ((predictions[0] + 1) * 127.5).astype(np.int)
        predictions[1] = ((predictions[1] + 1) * 127.5).astype(np.int)
        #color_images = batch[1]['color_output']

        fig, ax = plt.subplots(4, 4)

        for i in range(4):
            ax[i, 0].imshow(original_images[i])
            ax[i, 1].imshow(color_images[i])
            ax[i, 2].imshow(predictions[0][i])
            ax[i, 3].imshow(np.squeeze(predictions[1][i]))

        plt.savefig(os.path.join(self.save_path, "images/plot-epoch-{}.png".format(epoch_index)))
        plt.clf()
        plt.close(fig)

    def on_train_batch_end(self, batch_index, logs=None):
        """
        batch = self.sequence.__getitem__(0)
        predictions = self.model.predict(batch[0]['input_image'])

        fig, ax = plt.subplots(len(batch[0]['input_image']), 2)

        for i in range(len(batch[0]['input_image'])):
            ax[i, 0].imshow(batch[1]['color_output'][i])
            ax[i, 1].imshow(predictions[0][i])

        plt.savefig(os.path.join(self.save_path, "images/plot-epoch-{}.png".format(batch_index)))
        """
        pass


def Generator():

    """Auto-encoder model for latent-pose reconstruction.
    # Arguments
        input_shape: List of integers, indicating the initial tensor shape.
        latent_dimension: Integer, value of the latent vector dimension.
        mode: String {`full`, `encoder`, `decoder`}.
            If `full` both encoder-decoder parts are returned as a single model
            If `encoder` only the encoder part is returned as a single model
            If `decoder` only the decoder part is returned as a single model
        dropout_rate: Float between [0, 1], indicating the dropout rate of the
            latent vector.
    """
    bn_axis = 3

    input = Input((128, 128, 3), name='input_image')

    # First layer of the encoder
    e1_1 = Conv2D(64, (5, 5), strides=(2, 2), padding='same', name='encoder_conv2D_1_1')(input)
    e1_1 = BatchNormalization(bn_axis)(e1_1)
    e1_1 = LeakyReLU()(e1_1)

    e1_2 = Conv2D(64, (5, 5), strides=(2, 2), padding='same', name='encoder_conv2D_1_2')(input)
    e1_2 = BatchNormalization(bn_axis)(e1_2)
    e1_1 = LeakyReLU()(e1_1)

    e1 = Concatenate()([e1_1, e1_2])

    # Second layer of the encoder
    e2_1 = Conv2D(128, (5, 5), strides=(2, 2), padding='same', name='encoder_conv2D_2_1')(e1)
    e2_1 = BatchNormalization(bn_axis)(e2_1)
    e2_1 = LeakyReLU()(e2_1)

    e2_2 = Conv2D(128, (5, 5), strides=(2, 2), padding='same', name='encoder_conv2D_2_2')(e1)
    e2_2 = BatchNormalization(bn_axis)(e2_2)
    e2_2 = LeakyReLU()(e2_2)

    e2 = Concatenate()([e2_1, e2_2])

    # Third layer of the encoder
    e3_1 = Conv2D(128, (5, 5), strides=(2, 2), padding='same', name='encoder_conv2D_3_1')(e2)
    e3_1 = BatchNormalization(bn_axis)(e3_1)
    e3_1 = LeakyReLU()(e3_1)

    e3_2 = Conv2D(128, (5, 5), strides=(2, 2), padding='same', name='encoder_conv2D_3_2')(e2)
    e3_2 = BatchNormalization(bn_axis)(e3_2)
    e3_2 = LeakyReLU()(e3_2)

    e3 = Concatenate()([e3_1, e3_2])

    # Fourth layer of the encoder
    e4_1 = Conv2D(256, (5, 5), strides=(2, 2), padding='same', name='encoder_conv2D_4_1')(e3)
    e4_1 = BatchNormalization(bn_axis)(e4_1)
    e4_1 = LeakyReLU()(e4_1)

    e4_2 = Conv2D(256, (5, 5), strides=(2, 2), padding='same', name='encoder_conv2D_4_2')(e3)
    e4_2 = BatchNormalization(bn_axis)(e4_2)
    e4_2 = LeakyReLU()(e4_2)

    e4 = Concatenate()([e4_1, e4_2])

    # Latent dimension
    x = Flatten()(e4)
    x = Dense(256)(x)
    x = Dense(8*8*256)(x)
    x = Reshape((8, 8, 256))(x)

    # First layer of the decoder
    d1_1 = Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', name='decoder_conv2D_1_1')(x)
    d1_1 = BatchNormalization(bn_axis)(d1_1)
    d1_1 = LeakyReLU()(d1_1)

    d1 = Concatenate()([d1_1, e3_2])

    # Second layer of the decoder
    d2_1 = Conv2D(256, (5, 5), strides=(1, 1), padding='same', name='decoder_conv2D_2_1')(d1)
    d2_1 = BatchNormalization(bn_axis)(d2_1)
    d2_1 = LeakyReLU()(d2_1)

    d2_2 = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', name='decoder_conv2D_2_2')(d2_1)
    d2_2 = BatchNormalization(bn_axis)(d2_2)
    d2_2 = LeakyReLU()(d2_2)

    d2 = Concatenate()([d2_2, e2_2])

    # Third layer of the decoder
    d3_1 = Conv2D(256, (5, 5), strides=(1, 1), padding='same', name='decoder_conv2D_3_1')(d2)
    d3_1 = BatchNormalization(bn_axis)(d3_1)
    d3_1 = LeakyReLU()(d3_1)

    d3_2 = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', name='decoder_conv2D_3_2')(d3_1)
    d3_2 = BatchNormalization(bn_axis)(d3_2)
    d3_2 = LeakyReLU()(d3_2)

    d3 = Concatenate()([d3_2, e1_2])

    # Fourth layer
    d4_1 = Conv2D(128, (5, 5), strides=(1, 1), padding='same', name='decoder_conv2D_4_1')(d3)
    d4_1 = BatchNormalization(bn_axis)(d4_1)
    d4_1 = LeakyReLU()(d4_1)

    # Define the two outputs
    color_output = Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same')(d4_1)
    color_output = Activation('tanh', name='color_output')(color_output)

    error_output = Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same')(d4_1)
    error_output = Activation('sigmoid', name='error_output')(error_output)

    # Define model
    model = Model(inputs=[input], outputs=[color_output, error_output])
    #model.compile(optimizer='adam', loss=transformer_loss)
    #model.summary()
    return model


def Discriminator():
    bn_axis = 3

    input = Input((128, 128, 3), name='input_image')

    # First layer of the discriminator
    d1 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', name='discriminator_conv2D_1_1')(input)
    d1 = BatchNormalization(bn_axis)(d1)
    d1 = LeakyReLU(0.2)(d1)

    # Second layer of the discriminator
    d2 = Conv2D(128, (3, 3), strides=(2, 2), padding='same', name='discriminator_conv2D_2_1')(d1)
    d2 = BatchNormalization(bn_axis)(d2)
    d2 = LeakyReLU(0.2)(d2)

    # Third layer of the discriminator
    d3 = Conv2D(256, (3, 3), strides=(2, 2), padding='same', name='discriminator_conv2D_3_1')(d2)
    d3 = BatchNormalization(bn_axis)(d3)
    d3 = LeakyReLU(0.2)(d3)

    # Fourth layer of the discriminator
    d4 = Conv2D(512, (3, 3), strides=(2, 2), padding='same', name='discriminator_conv2D_4_1')(d3)
    d4 = BatchNormalization(bn_axis)(d4)
    d4 = LeakyReLU(0.2)(d4)

    # Fifth layer of the discriminator
    d5 = Conv2D(512, (3, 3), strides=(2, 2), padding='same', name='discriminator_conv2D_5_1')(d4)
    d5 = BatchNormalization(bn_axis)(d5)
    d5 = LeakyReLU(0.2)(d5)

    # Sixth layer of the discriminator
    d6 = Conv2D(512, (3, 3), strides=(2, 2), padding='same', name='discriminator_conv2D_6_1')(d5)
    d6 = BatchNormalization(bn_axis)(d6)
    d6 = LeakyReLU(0.2)(d6)

    # Seventh layer of the discriminator
    d7 = Conv2D(512, (3, 3), strides=(2, 2), padding='same', name='discriminator_conv2D_7_1')(d6)
    d7 = BatchNormalization(bn_axis)(d7)
    d7 = LeakyReLU(0.2)(d7)

    flatten = Flatten()(d7)
    output = Dense(1, activation='sigmoid', name='discriminator_output')(flatten)
    discriminator_model = Model(inputs=input, outputs=[output])
    return discriminator_model


class GAN():
    def __init__(self):
        pass


if __name__ == '__main__':
    disc = Discriminator()
    print(disc.summary())