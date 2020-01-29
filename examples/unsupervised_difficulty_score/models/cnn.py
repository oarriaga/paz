from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Flatten, Input
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D


def CNN(input_shape, num_classes):
    inputs = Input(input_shape, name='image')
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax', name='label')(x)
    model = Model(inputs, outputs, name='CNN')
    return model
