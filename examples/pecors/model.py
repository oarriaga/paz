import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Input, LeakyReLU, Concatenate, \
    Flatten, Dense, Reshape, Conv2DTranspose, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback

import neptune

import matplotlib.pyplot as plt

"""
Punishes the loss of the circle harder
"""
def pecors_loss(circle_image, predicted_circle_image):
    min_loss = tf.float32.max
    beta = 25

    # Calculate masks for the circle and the background
    mask_circle = tf.repeat(tf.expand_dims(tf.math.reduce_max(tf.math.ceil(circle_image), axis=-1), axis=-1),
                            repeats=3, axis=-1)
    mask_background = tf.ones(tf.shape(mask_circle)) - mask_circle

    # Calculate the difference between the real and predicted images including the mask
    diff_object = tf.math.square(predicted_circle_image * mask_circle - circle_image * mask_circle)
    diff_background = tf.math.square(predicted_circle_image * mask_background - circle_image * mask_background)

    # Calculate the total loss
    num_pixels = tf.math.reduce_prod(tf.shape(circle_image)[1:3])
    loss_circle = tf.cast((1 / num_pixels), dtype=tf.float32) * (beta * tf.math.reduce_sum(diff_object, axis=[1, 2, 3])
                                                                 + tf.math.reduce_sum(diff_background, axis=[1, 2, 3]))

    return loss_circle


def Pecors():
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

    # Circle output: shows the vector of the top of the object
    color_output = Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same')(d4_1)
    color_output = Activation('sigmoid', name='circle_output')(color_output)

    # Depth output: position of the center of the object and the depth
    depth_output = Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same')(d4_1)
    depth_output = Activation('sigmoid', name='depth_output')(depth_output)

    # Define model
    model = Model(inputs=[input], outputs=[color_output, depth_output])
    #model.compile(optimizer='adam', loss=transformer_loss)
    #model.summary()
    return model


class NeptuneLogger(Callback):

    def __init__(self, model, log_interval, save_path):
        self.model = model
        self.log_interval = log_interval
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs={}):
        for log_name, log_value in logs.items():
            neptune.log_metric(log_name, log_value)

        if epoch%self.log_interval == 0:
            #self.model.save(os.path.join(self.save_path, 'dope_model_epoch_{}.pkl'.format(epoch)))
            neptune.log_artifact('pecors_model_epoch_{}.pkl'.format(epoch))


class PlotImagesCallback(Callback):
    def __init__(self, model, sequence, neptune_logging=False):
        self.model = model
        self.sequence = sequence
        self.neptune_logging = neptune_logging

    def on_epoch_end(self, epoch_index, logs=None):
        batch = self.sequence.__getitem__(0)
        predictions = self.model.predict(batch[0]['input_image'])
        original_images = batch[0]['input_image']

        # num_cols: original image, real circle image, predicted circle image
        num_rows = 5
        num_cols = 5
        fig, ax = plt.subplots(num_rows, num_cols)
        fig.set_size_inches(12, 8)

        for i in range(num_rows):
            for j in range(num_cols):
                ax[i, j].get_xaxis().set_visible(False)
                ax[i, j].get_yaxis().set_visible(False)

        # Show original image in the first row
        ax[0, 0].set_title("Input image")
        for i in range(num_rows):
            ax[i, 0].imshow(original_images[i])

        # Show real circle images
        ax[0, 1].set_title("Real circle images")
        for i in range(num_rows):
            ax[i, 1].imshow(batch[1]["circle_output"][i])

        # Show the predicted circle images
        ax[0, 2].set_title("Predicted circle images")
        for i in range(num_rows):
            ax[i, 2].imshow(predictions[0][i])
            
        # Show real depth images
        ax[0, 3].set_title("Real depth images")
        for i in range(num_rows):
            ax[i, 3].imshow(batch[1]["depth_output"][i])

        # Show the predicted depth images
        ax[0, 4].set_title("Predicted depth images")
        for i in range(num_rows):
            ax[i, 4].imshow(predictions[1][i])

        #plt.tight_layout()
        fig.subplots_adjust(hspace=0.5)

        if self.neptune_logging:
            neptune.log_image('plot', fig, image_name="epoch_{}.png".format(epoch_index))
        else:
            plt.savefig("images/plot-epoch-{}.png".format(epoch_index))

        #plt.show()

        plt.clf()
        plt.close(fig)
