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


def PoseNet(segmented_image):
    X_1 = Conv2D(64, 3, activation='relu', padding='same',
                 name='conv2_1')(segmented_image)
    X_2 = Conv2D(64, 3, activation='relu', padding='same', name='conv2_2')(X_1)
    X_3 = MaxPooling2D(pool_size=(2, 2))(X_2)
    X_4 = Conv2D(128, 3, activation='relu', padding='same', name='conv2_3')(X_3)
    X_5 = Conv2D(128, 3, activation='relu', padding='same', name='conv2_4')(X_4)
    X_6 = MaxPooling2D(pool_size=(2, 2))(X_5)
    X_7 = Conv2D(256, 3, activation='relu', padding='same', name='conv2_5')(X_6)
    X_8 = Conv2D(256, 3, activation='relu', padding='same', name='conv2_6')(X_7)
    X_9 = Conv2D(256, 3, activation='relu', padding='same', name='conv2_7')(X_8)
    X_10 = Conv2D(256, 3, activation='relu', padding='same',
                  name='conv2_8')(X_9)
    X_11 = MaxPooling2D(pool_size=(2, 2))(X_10)
    X_12 = Conv2D(512, 3, activation='relu', padding='same',
                  name='conv2_9')(X_11)
    X_13 = Conv2D(512, 3, activation='relu', padding='same',
                  name='conv2_10')(X_12)
    X_14 = Conv2D(512, 3, activation='relu', padding='same',
                  name='conv2_11')(X_13)
    X_15 = Conv2D(512, 3, activation='relu', padding='same',
                  name='conv2_12')(X_14)
    X_16 = Conv2D(512, 3, activation='relu', padding='same',
                  name='conv2_13')(X_15)
    X_17 = Conv2D(21, 1, padding='same', activation=None, name='conv2_14')(X_16)

    X_18 = Concatenate(axis=-1)([X_16, X_17])

    X_19 = Conv2D(128, 7, activation='relu', padding='same',
                  name='conv2_15')(X_18)
    X_20 = Conv2D(128, 7, activation='relu', padding='same',
                  name='conv2_16')(X_19)
    X_21 = Conv2D(128, 7, activation='relu', padding='same',
                  name='conv2_17')(X_20)
    X_22 = Conv2D(128, 7, activation='relu', padding='same',
                  name='conv2_18')(X_21)
    X_23 = Conv2D(128, 7, activation='relu', padding='same',
                  name='conv2_19')(X_22)
    X_24 = Conv2D(21, 1, padding='same', activation=None, name='conv2_20')(X_23)

    X_25 = Concatenate(axis=-1)([X_16, X_17, X_24])

    X_26 = Conv2D(128, 7, activation='relu', padding='same',
                  name='conv2_21')(X_25)
    X_27 = Conv2D(128, 7, activation='relu', padding='same',
                  name='conv2_22')(X_26)
    X_28 = Conv2D(128, 7, activation='relu', padding='same',
                  name='conv2_23')(X_27)
    X_29 = Conv2D(128, 7, activation='relu', padding='same',
                  name='conv2_24')(X_28)
    X_30 = Conv2D(128, 7, activation='relu', padding='same',
                  name='conv2_25')(X_29)
    X_31 = Conv2D(21, 1, padding='same', activation=None, name='conv2_26')(X_30)

    return X_31


def PosePriorNet(score_maps, hand_side, P):
    batch_size = score_maps.shape[0]
    X_1 = Conv2D(32, 3, activation='relu', padding='same',
                 name='conv3_1')(score_maps)
    X_2 = Conv2D(32, 3, activation='relu', padding='same', strides=2,
                 name='conv3_2')(X_1)
    X_3 = Conv2D(64, 3, activation='relu', padding='same',
                 name='conv3_3')(X_2)
    X_4 = Conv2D(64, 3, activation='relu', padding='same', strides=2,
                 name='conv3_4')(X_3)
    X_5 = Conv2D(128, 3, activation='relu', padding='same',
                 name='conv3_5')(X_4)
    X_6 = Conv2D(128, 3, padding='same', activation=None, strides=2,
                 name='conv3_6')(X_5)

    X_7 = tf.reshape(X_6, [batch_size, -1])  # this is Bx2048
    X_7 = tf.concat([X_7, hand_side], 1)

    X_8 = Dense(512, activation='relu')(X_7)
    X_8 = Dropout(rate=0.2)(X_8)

    X_9 = Dense(512, activation='relu')(X_8)
    X_9 = Dropout(rate=0.2)(X_9)

    X_10 = Dense(P)(X_9)
    return X_10


if __name__ == '__main__':
    pass
