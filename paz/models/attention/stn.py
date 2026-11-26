import keras
from keras.layers import Input, Activation, MaxPool2D, Flatten, Layer
from keras.layers import Conv2D, Dense
import jax
import paz


def batch_transform(images, matrices):
    return jax.vmap(paz.image.affine_transform)(images, matrices)


def batch_resize(images, shape):
    return jax.vmap(paz.lock(paz.image.resize, shape, "bilinear"))(images)


class BilinearInterpolation(Layer):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def call(self, images_and_transforms):
        images, affine_transforms = images_and_transforms
        images = batch_transform(images, affine_transforms)
        images = paz.lock(batch_resize, self.shape)(images)
        return images


class ToSE2(Layer):
    def __init__(self):
        super().__init__()

    def call(self, rotation_and_translation):
        rotation, translation = rotation_and_translation
        rotation = rotation.reshape((-1, 2, 2))
        return jax.vmap(paz.SE2.to_affine_matrix)(rotation, translation)


def identity(shape, dtype):
    return keras.ops.identity(2, dtype=dtype).flatten()


def STN(input_shape=(40, 40, 1), interpolation_size=(30, 30), num_classes=10):
    image = Input(shape=input_shape)
    x = MaxPool2D(pool_size=(2, 2))(image)
    x = Conv2D(20, (5, 5))(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Conv2D(20, (5, 5))(x)
    x = Flatten()(x)
    x = Dense(10)(x)
    x = Activation("relu")(x)
    r = Dense(4, kernel_initializer="zeros", bias_initializer=identity)(x)
    t = Dense(2, kernel_initializer="zeros", bias_initializer="random_normal")(
        x
    )
    a = ToSE2()([r, t])
    interpolated_image = BilinearInterpolation(interpolation_size)([image, a])
    x = Conv2D(32, (3, 3), padding="same")(interpolated_image)
    x = Activation("relu")(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Conv2D(32, (3, 3))(x)
    x = Activation("relu")(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(32)(x)
    x = Activation("relu")(x)
    x = Dense(num_classes)(x)
    x = Activation("softmax", name="label")(x)
    return keras.Model(image, x, name="STN")
