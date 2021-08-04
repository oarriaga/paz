import matplotlib.pyplot as plt
import os
import neptune

from tensorflow.keras.layers import Conv2D, Activation, UpSampling2D, Dense
from tensorflow.keras.layers import Dropout, Input, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback


class PlotImagesCallback(Callback):
    def __init__(self, model, sequence, save_path, batch_size, neptune_logging=False):
        self.save_path = save_path
        self.model = model
        self.sequence = sequence
        self.neptune_logging = neptune_logging
        self.batch_size = batch_size

    def on_epoch_end(self, epoch_index, epoch_logs):
        sequence_iterator = self.sequence.__iter__()
        batch = next(sequence_iterator)
        predictions = self.model.predict(batch[0]['input_image'])

        num_columns = 3
        num_samples = 4

        fig, ax = plt.subplots(4, num_columns)
        fig.set_size_inches(10, 6)

        cols = ["Input image", "Ground truth", "Predicted image"]

        for i in range(num_columns):
            ax[0, i].set_title(cols[i])
            for j in range(num_samples):
                ax[j, i].get_xaxis().set_visible(False)
                ax[j, i].get_yaxis().set_visible(False)

        for i in range(num_samples):
            ax[i, 0].imshow(batch[0]['input_image'][i])
            ax[i, 1].imshow(batch[1]['label_image'][i])
            ax[i, 2].imshow(predictions[i])

        if self.neptune_logging:
            neptune.log_image('plot', fig, image_name="epoch_{}.png".format(epoch_index))

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
            self.model.save(os.path.join(self.save_path, 'implicit_orientation_{}.h5'.format(epoch)))


def AutoEncoder(input_shape, latent_dimension=128, mode='full',
                dropout_rate=None):

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

    if mode not in ['full', 'encoder', 'decoder']:
        raise ValueError('Invalid mode.')

    i = Input(input_shape, name='input_image')
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same', name='conv2D_1')(i)
    x = Activation('relu', name='relu_1')(x)
    x = Conv2D(64, (3, 3), strides=(2, 2), padding='same', name='conv2D_2')(x)
    x = Activation('relu', name='relu_2')(x)
    x = Conv2D(128, (3, 3), strides=(2, 2), padding='same', name='conv2D_3')(x)
    x = Activation('relu', name='relu_3')(x)
    x = Conv2D(256, (3, 3), strides=(2, 2), padding='same', name='conv2D_4')(x)
    x = Activation('relu', name='relu_4')(x)
    x = Flatten(name='flatten_1')(x)

    z = Dense(latent_dimension, name='latent_vector')(x)
    if dropout_rate is not None:
        z = Dropout(dropout_rate, name='latent_vector_dropout')(z)

    if mode == 'decoder':
        z = Input(shape=(latent_dimension, ), name='input')
    x = Dense(8 * 8 * 256, name='dense_1')(z)
    x = Reshape((8, 8, 256), name='reshape_1')(x)
    x = UpSampling2D((2, 2), name='upsample_1')(x)
    x = Conv2D(128, (3, 3), padding='same', name='conv2D_5')(x)
    x = Activation('relu', name='relu_5')(x)
    x = UpSampling2D((2, 2), name='upsample_2')(x)
    x = Conv2D(64, (3, 3), padding='same', name='conv2D_6')(x)
    x = Activation('relu', name='relu_6')(x)
    x = UpSampling2D((2, 2), name='upsample_3')(x)
    x = Conv2D(32, (3, 3), padding='same', name='conv2D_7')(x)
    x = Activation('relu', name='relu_7')(x)
    x = UpSampling2D((2, 2), name='upsample_4')(x)
    x = Conv2D(input_shape[-1], (3, 3), padding='same', name='conv2D_8')(x)
    output_tensor = Activation('sigmoid', name='label_image')(x)
    base_name = 'SimpleAutoencoder' + str(latent_dimension)
    if dropout_rate is not None:
        base_name = base_name + 'DRP_' + str(dropout_rate)

    if mode == 'encoder':
        name = base_name + '-encoder'
        model = Model(i, z, name=name)

    elif mode == 'decoder':
        name = base_name + '-decoder'
        model = Model(z, output_tensor, name=name)

    elif mode == 'full':
        model = Model(i, output_tensor, name=base_name)

    return model
