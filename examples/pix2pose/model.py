import numpy as np
import sys
import glob
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import neptune

from tensorflow.keras.layers import Conv2D, Activation, UpSampling2D, Dense, Conv2DTranspose
from tensorflow.keras.layers import Dropout, Input, Flatten, Reshape, LeakyReLU, BatchNormalization, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
import tensorflow as tf

from scenes import SingleView
from pipelines import DepthImageGenerator
from paz.abstract.sequence import GeneratingSequencePix2Pose

#tf.config.run_functions_eagerly(True)


def transformer_loss(real_depth_image, predicted_depth_image):
    print("shape of real_depth_image: {}".format(real_depth_image.numpy().shape))
    print("shape of predicted_depth_image: {}".format(predicted_depth_image.numpy().shape))

    plt.imshow(predicted_depth_image.numpy()[0])
    plt.show()

    plt.imshow(predicted_depth_image.numpy()[1])
    plt.show()
    #print(predicted_depth_image.numpy()[0])
    return K.mean(K.square(predicted_depth_image - real_depth_image), axis=-1)


def loss_color_wrapped(rotation_matrices):
    def loss_color_unwrapped(color_image, predicted_color_image):
        min_loss = tf.float32.max

        # Iterate over all possible rotations
        for rotation_matrix in rotation_matrices:

            real_color_image = tf.identity(color_image)

            # Bring the image in the range between 0 and 1
            real_color_image = (real_color_image+1)*0.5

            # Calculate masks for the object and the background (they are independent of the rotation)
            mask_object = tf.repeat(tf.expand_dims(tf.math.reduce_max(tf.math.ceil(real_color_image), axis=-1), axis=-1), repeats=3, axis=-1)
            mask_background = tf.ones(tf.shape(mask_object)) - mask_object

            # Add a small epsilon value to avoid the discontinuity problem
            real_color_image = real_color_image + tf.ones_like(real_color_image) * 0.0001

            # Rotate the object
            real_color_image = tf.einsum('ij,mklj->mkli', tf.convert_to_tensor(np.array(rotation_matrix), dtype=tf.float32), real_color_image)
            real_color_image = tf.where(tf.math.less(real_color_image, 0), tf.ones_like(real_color_image) + real_color_image, real_color_image)

            real_color_image = real_color_image*mask_object

            # Bring the image again in the range between -1 and 1
            real_color_image = (real_color_image*2)-1

            # Get the number of pixels
            num_pixels = tf.math.reduce_prod(tf.shape(real_color_image)[1:3])
            beta = 3

            # Calculate the difference between the real and predicted images including the mask
            diff_object = tf.math.abs(predicted_color_image*mask_object - real_color_image*mask_object)
            diff_background = tf.math.abs(predicted_color_image*mask_background - real_color_image*mask_background)

            # Calculate the total loss
            loss_colors = tf.cast((1/num_pixels), dtype=tf.float32)*(beta*tf.math.reduce_sum(diff_object, axis=[1, 2, 3]) + tf.math.reduce_sum(diff_background, axis=[1, 2, 3]))
            min_loss = tf.math.minimum(loss_colors, min_loss)
        return min_loss

    return loss_color_unwrapped


def loss_color(color_image, predicted_color_image):

        real_color_image = tf.identity(color_image)

        # Calculate masks for the object and the background (they are independent of the rotation)
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

        return loss_colors


def loss_error(real_error_image, predicted_error_image):

    # Get the number of pixels
    num_pixels = tf.math.reduce_prod(tf.shape(real_error_image)[1:3])
    loss_error = tf.cast((1/num_pixels), dtype=tf.float32)*(tf.math.reduce_sum(tf.math.square(predicted_error_image - tf.clip_by_value(tf.math.abs(real_error_image), tf.float32.min, 1.)), axis=[1, 2, 3]))

    return loss_error


class PlotImagesCallback(Callback):
    def __init__(self, model, sequence, save_path, obj_path, image_size, multipleHypotheses, neptune_logging=False):
        self.save_path = save_path
        self.model = model
        self.sequence = sequence
        self.neptune_logging = neptune_logging
        self.obj_path = obj_path
        self.image_size = image_size
        self.multipleHypotheses = multipleHypotheses

    def on_epoch_end(self, epoch_index, logs=None):
        sequence_iterator = self.sequence.__iter__()
        batch = next(sequence_iterator)
        predictions = self.model.predict(batch[0]['input_image'])

        original_images = (batch[0]['input_image'] * 255).astype(np.int)
        color_images = ((batch[1]['color_output'] + 1) * 127.5).astype(np.int)
        color_images = color_images.astype(np.float)/255.

        for color_output_layer_name in self.multipleHypotheses.names_hypotheses_layers['color_output']:
            predictions[color_output_layer_name] = ((predictions[color_output_layer_name] + 1) * 127.5).astype(np.int)

        for error_output_layer_name in self.multipleHypotheses.names_hypotheses_layers['error_output']:
            predictions[error_output_layer_name] = ((predictions[error_output_layer_name] + 1) * 127.5).astype(np.int)

        num_columns = self.multipleHypotheses.M
        row_list = list(range(0, 6, 3))

        # Three time the number of rows:
        # First row: Original image, real color image
        # Second row: All predicted color outputs
        # Third row: All predicted error outputs
        fig, ax = plt.subplots(len(row_list)*3, num_columns)
        fig.set_size_inches(10, 6)

        for num_row in range(len(row_list)*3):
            for num_column in range(num_columns):
                ax[num_row, num_column].get_xaxis().set_visible(False)
                ax[num_row, num_column].get_yaxis().set_visible(False)

        for i, row in enumerate(row_list):
            # First row: original image, real color image, real error image
            ax[row, 0].imshow(original_images[i])
            ax[row, 1].imshow(color_images[i])

            # Second row: All predicted color outputs
            for num_color_layer, color_output_layer_name in enumerate(self.multipleHypotheses.names_hypotheses_layers['color_output']):
                ax[row+1, num_color_layer].imshow(predictions[color_output_layer_name][i])

            # Third row: All predicted error outputs
            for num_error_layer, error_output_layer_name in enumerate(self.multipleHypotheses.names_hypotheses_layers['error_output']):
                ax[row+2, num_error_layer].imshow(predictions[error_output_layer_name][i])

        plt.tight_layout()
        plt.show()
        #plt.savefig(os.path.join(self.save_path, "images/plot-epoch-{}.png".format(epoch_index)))

        #if self.neptune_logging:
        #    neptune.log_image('plot', fig, image_name="epoch_{}.png".format(epoch_index))

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
            self.model.save(os.path.join(self.save_path, 'pix2pose_dcgan_{}.h5'.format(epoch)))
            neptune.log_artifact('pix2pose_dcgan_{}.h5'.format(epoch))


def Generator(multipleHypotheses=None):
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
    if multipleHypotheses:
        output_layers_list = multipleHypotheses.multiple_hypotheses_output(d4_1, {
            'color_output': Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', activation='tanh'),
            'error_output': Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='sigmoid')})
        print(output_layers_list)
        model = Model(inputs=[input], outputs=output_layers_list)
    else:
        color_output = Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same')(d4_1)
        color_output = Activation('tanh', name='color_output')(color_output)

        error_output = Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same')(d4_1)
        error_output = Activation('sigmoid', name='error_output')(error_output)

        # Define model
        model = Model(inputs=[input], outputs=[color_output, error_output])

    #model.compile(optimizer='adam', loss=transformer_loss)
    model.summary()
    plot_model(model, "model.png")
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


if __name__ == '__main__':
    Generator()
