from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization, ReLU, Add, UpSampling2D, Concatenate
from tensorflow.keras.utils import plot_model

def create_upsampling_block(input_upsampling_layer_01, input_downsampling_layer_01,
                            input_downsampling_layer_03, input_downsampling_layer_05):
    # First upsampling block
    x = UpSampling2D(interpolation='bilinear')(input_upsampling_layer_01)
    x = Concatenate()([x, input_downsampling_layer_05])
    x = Conv2D(128, (3, 3), padding='same')(x)

    # Second upsampling block
    x = UpSampling2D(interpolation='bilinear')(x)
    x = Concatenate()([x, input_downsampling_layer_03])
    x = Conv2D(64, (3, 3), padding='same')(x)

    # Third upsampling block
    x = UpSampling2D(interpolation='bilinear')(x)
    x = Concatenate()([x, input_downsampling_layer_01])
    x = Conv2D(64, (3, 3), padding='same')(x)

    # Fourth upsampling block
    x = UpSampling2D(interpolation='bilinear')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)

    return x


def DPOD(input_shape, num_objects, num_colors):
    i = Input(input_shape, name='input_image')
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(i)
    x = BatchNormalization()(x)
    input_maxpool = ReLU()(x)
    input_downsampling_layer_01 = MaxPool2D(pool_size=(2, 2))(input_maxpool)

    # First downsampling layer
    x = Conv2D(64, (3, 3), padding='same')(input_downsampling_layer_01)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, input_downsampling_layer_01])
    input_downsampling_layer_02 = ReLU()(x)

    # Second downsampling layer
    x = Conv2D(64, (3, 3), padding='same')(input_downsampling_layer_02)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, input_downsampling_layer_02])
    input_downsampling_layer_03 = ReLU()(x)

    # Third downsampling layer
    x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(input_downsampling_layer_03)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    input_downsampling_layer_03_changed = Conv2D(128, (1, 1), strides=(2, 2), padding='same')(input_downsampling_layer_03)
    x = Add()([x, input_downsampling_layer_03_changed])
    input_downsampling_layer_04 = ReLU()(x)

    # Fourth downsampling layer
    x = Conv2D(128, (3, 3), padding='same')(input_downsampling_layer_04)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, input_downsampling_layer_04])
    input_downsampling_layer_05 = ReLU()(x)

    # Fifth downsampling layer
    x = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(input_downsampling_layer_05)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    input_downsampling_layer_05_changed = Conv2D(256, (1, 1), strides=(2, 2), padding='same')(input_downsampling_layer_05)
    x = Add()([x, input_downsampling_layer_05_changed])
    input_downsampling_layer_06 = ReLU()(x)

    # Sixth downsampling layer
    x = Conv2D(256, (3, 3), padding='same')(input_downsampling_layer_06)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, input_downsampling_layer_06])
    input_upsampling_layer_01 = ReLU()(x)

    u_map_input = create_upsampling_block(input_upsampling_layer_01, input_maxpool, input_downsampling_layer_03, input_downsampling_layer_05)
    v_map_input = create_upsampling_block(input_upsampling_layer_01, input_maxpool, input_downsampling_layer_03, input_downsampling_layer_05)
    id_mask_input = create_upsampling_block(input_upsampling_layer_01, input_maxpool, input_downsampling_layer_03, input_downsampling_layer_05)

    u_map_output = Conv2D(num_colors, (3,3), padding='same', name="u_map_output")(u_map_input)
    v_map_output = Conv2D(num_colors, (3, 3), padding='same', name="v_map_output")(v_map_input)
    id_mask_output = Conv2D(num_objects, (3, 3), padding='same', name="id_mask_output")(id_mask_input)

    model = Model(inputs=[i], outputs=[u_map_output, v_map_output, id_mask_output])
    model.summary()
    plot_model(model, to_file="dpod_model.png", show_shapes=True)

    return model


if __name__ == "__main__":
    DPOD((320, 320, 3), num_objects=2, num_colors=255)