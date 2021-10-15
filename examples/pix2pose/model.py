import numpy as np

from tensorflow.keras.layers import Conv2D, Activation, UpSampling2D, Dense, Conv2DTranspose, Dropout, Input, Flatten, Reshape, LeakyReLU, BatchNormalization, Concatenate
from tensorflow.keras.models import Model
import tensorflow as tf


def loss_color_wrapped(rotation_matrices):
    def loss_color_unwrapped(color_image, predicted_color_image):
        min_loss = tf.float32.max

        # Bring the image in the range between 0 and 1
        color_image = (color_image + 1) * 0.5

        # Calculate masks for the object and the background (they are independent of the rotation)
        mask_object = tf.repeat(tf.expand_dims(tf.math.reduce_max(tf.math.ceil(color_image), axis=-1), axis=-1),
                                repeats=3, axis=-1)
        mask_background = tf.ones(tf.shape(mask_object)) - mask_object

        # Bring the image again in the range between -1 and 1
        color_image = (color_image * 2) - 1

        # Iterate over all possible rotations
        for rotation_matrix in rotation_matrices:

            real_color_image = tf.identity(color_image)

            # Add a small epsilon value to avoid the discontinuity problem
            real_color_image = real_color_image + tf.ones_like(real_color_image) * 0.0001

            # Rotate the object
            real_color_image = tf.einsum('ij,mklj->mkli', tf.convert_to_tensor(np.array(rotation_matrix), dtype=tf.float32), real_color_image)
            #real_color_image = tf.where(tf.math.less(real_color_image, 0), tf.ones_like(real_color_image) + real_color_image, real_color_image)

            # Set the background to be all -1
            real_color_image *= mask_object
            real_color_image += (mask_background*tf.constant(-1.))

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


def loss_error(real_error_image, predicted_error_image):
    # Get the number of pixels
    num_pixels = tf.math.reduce_prod(tf.shape(real_error_image)[1:3])
    loss_error = tf.cast((1/num_pixels), dtype=tf.float32)*(tf.math.reduce_sum(tf.math.square(predicted_error_image - tf.clip_by_value(tf.math.abs(real_error_image), tf.float32.min, 1.)), axis=[1, 2, 3]))

    return loss_error


def Generator():
    bn_axis = 3

    input = Input((128, 128, 3), name='input_image')

    # First layer of the encoder
    e1_1 = Conv2D(64, (5, 5), strides=(2, 2), padding='same', name='encoder_conv2D_1_1')(input)
    e1_1 = BatchNormalization(bn_axis)(e1_1)
    e1_1 = LeakyReLU()(e1_1)

    e1_2 = Conv2D(64, (5, 5), strides=(2, 2), padding='same', name='encoder_conv2D_1_2')(input)
    e1_2 = BatchNormalization(bn_axis)(e1_2)
    e1_2 = LeakyReLU()(e1_2)

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