from keras import Model
from keras.layers import (
    Conv2D,
    Activation,
    UpSampling2D,
    Dense,
    Input,
    Flatten,
    Reshape,
)


def AutoEncoder(input_shape, latent_dimension=128):
    """Convolutional autoencoder for implicit orientation learning.

    The encoder maps a rendered object view to a latent vector; the decoder
    reconstructs the clean view. After training, `extract_encoder` returns the
    encoder used to build the orientation codebook. Input is `128x128`.
    """
    image = Input(input_shape, name="input_image")
    x = encode(image)
    latent = Dense(latent_dimension, name="latent_vector")(x)
    reconstruction = decode(latent, input_shape[-1])
    name = "Autoencoder" + str(latent_dimension)
    return Model(image, reconstruction, name=name)


def encode(x):
    for num_kernels in [32, 64, 128, 256]:
        x = Conv2D(num_kernels, 3, strides=2, padding="same")(x)
        x = Activation("relu")(x)
    return Flatten()(x)


def decode(latent, num_channels):
    x = Dense(8 * 8 * 256)(latent)
    x = Reshape((8, 8, 256))(x)
    for num_kernels in [128, 64, 32]:
        x = UpSampling2D(2)(x)
        x = Conv2D(num_kernels, 3, padding="same")(x)
        x = Activation("relu")(x)
    x = UpSampling2D(2)(x)
    x = Conv2D(num_channels, 3, padding="same")(x)
    return Activation("sigmoid", name="label_image")(x)


def extract_encoder(model):
    latent = model.get_layer("latent_vector").output
    return Model(model.input, latent, name="encoder")
