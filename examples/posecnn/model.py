from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Add, UpSampling2D, Dropout, Layer
from tensorflow.keras.models import Model

from tensorflow.keras.applications.vgg16 import VGG16


class LogSoftmax(Layer):
    def __init__(self, **kwargs):
        super(LogSoftmax, self).__init__(**kwargs)

    def call(self, inputs):
        print(inputs.shape)
        return inputs


def PoseCNN(num_classes=2):
    # Inspired by https://github.com/NVlabs/PoseCNN-PyTorch/blob/main/lib/networks/PoseCNN.py
    # VGG-16 backend
    model_vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    model_vgg16.summary()

    vgg16_first_intermediate_layer = model_vgg16.get_layer('block4_conv3').output
    vgg16_second_intermediate_layer = model_vgg16.get_layer('block5_conv3').output

    # First output for semantic segmentation
    x01 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(vgg16_first_intermediate_layer)

    x02 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(vgg16_second_intermediate_layer)
    x02 = UpSampling2D(size=(2, 2), interpolation='bilinear')(x02)

    add_layer = Add()([x01, x02])
    x = UpSampling2D(size=(8, 8), interpolation='bilinear')(add_layer)
    #x = Dropout()(x)
    semantic_segmentation_out = Conv2D(num_classes, kernel_size=(2, 2), padding='same', activation='relu')(x)
    log_softmax_out = LogSoftmax(name='log_softmax_out')(semantic_segmentation_out)

    model = Model(inputs=[model_vgg16.get_layer("input_1").input], outputs=[log_softmax_out])
    print(model.summary())

    return model


if __name__ == "__main__":
    poseCNN = PoseCNN()