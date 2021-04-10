import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D


def HandSegNet(image):
    X_1 = Conv2D(64, kernel_size=3, activation='relu', padding='same',
                 name='conv1_1')(image)
    X_2 = Conv2D(64, 3, activation='relu', padding='same', name='conv1_2')(X_1)
    X_3 = MaxPooling2D(pool_size=(2, 2))(X_2)
    X_4 = Conv2D(128, 3, padding='same', activation='relu', name='conv1_3')(X_3)
    X_5 = Conv2D(128, 3, padding='same', activation='relu', name='conv1_4')(X_4)
    X_6 = MaxPooling2D(pool_size=(2, 2))(X_5)
    X_7 = Conv2D(256, 3, padding='same', activation='relu', name='conv1_5')(X_6)
    X_8 = Conv2D(256, 3, padding='same', activation='relu', name='conv1_6')(X_7)
    X_9 = Conv2D(256, 3, padding='same', activation='relu', name='conv1_7')(X_8)
    X_10 = Conv2D(256, 3, padding='same', activation='relu',
                  name='conv1_8')(X_9)
    X_11 = MaxPooling2D(pool_size=(2, 2))(X_10)

    X_12 = Conv2D(512, 3, padding='same', activation='relu',
                  name='conv1_9')(X_11)
    X_13 = Conv2D(512, 3, padding='same', activation='relu',
                  name='conv1_10')(X_12)
    X_14 = Conv2D(512, 3, padding='same', activation='relu',
                  name='conv1_11')(X_13)
    X_15 = Conv2D(512, 3, padding='same', activation='relu',
                  name='conv1_12')(X_14)
    X_16 = Conv2D(512, 3, padding='same', activation='relu',
                  name='conv1_13')(X_15)
    X_17 = Conv2D(2, 1, padding='same', activation=None, name='conv1_14')(X_16)
    X_18 = UpSampling2D(size=(8, 8), interpolation='bilinear')(X_17)

    X = tf.argmax(X_18, axis=-1)
    X = tf.expand_dims(X, axis=-1)
    return X


if __name__ == '__main__':
    pass

