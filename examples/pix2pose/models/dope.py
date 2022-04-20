from tensorflow.keras.layers import Conv2D, Concatenate
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model


def build_stage(x, filters, kernel_shape, num_layers, activation, name):
    filter1, filter2 = filters
    print(num_layers)
    for layer_arg in range(num_layers):
        x = Conv2D(128, kernel_shape, padding='same', activation='relu')(x)
    x = Conv2D(filter1, 1, padding='same', activation='relu')(x)
    x = Conv2D(filter2, 1, padding='same', activation=activation, name=name)(x)
    return x


def build_belief_map(x, filters, kernel_shape, num_layers, stage_arg):
    name = '{}_stage_{}'.format('belief_maps', stage_arg)
    return build_stage(x, filters, kernel_shape, num_layers, 'sigmoid', name)


def build_affine_map(x, filters, kernel_shape, num_layers, stage_arg):
    name = '{}_stage_{}'.format('affine_maps', stage_arg)
    return build_stage(x, filters, kernel_shape, num_layers, 'linear', name)


def DOPE_VGG19(num_belief_maps=9, num_affine_maps=16,
               input_shape=(400, 400, 3), weights='imagenet', num_stages=1):

    backbone = VGG19(False, weights, input_shape=input_shape)
    stem = backbone.get_layer('block4_conv3').output
    stem = Conv2D(256, 3, padding='same', activation='relu')(stem)
    stage = stem = Conv2D(128, 3, padding='same', activation='relu')(stem)
    outputs = []
    belief_map = build_belief_map(stem, [512, num_belief_maps], 3, 3, 1)
    # affine_map = build_affine_map(stem, [512, num_affine_maps], 3, 3, 1)
    stage = Concatenate()([stem, belief_map])
    outputs.append(belief_map)
    if num_stages == 1:
        return Model(backbone.input, outputs, name='DOPE')

    for arg in range(2, num_stages + 1):
        belief_map = build_belief_map(stage, [128, num_belief_maps], 7, 6, arg)
        # affine_map = build_affine_map(stage, [128,num_belief_maps],7,6,arg)
        stage = Concatenate()([stem, belief_map])
        outputs.append(belief_map)
    return Model(backbone.input, outputs, name='DOPE')


if __name__ == "__main__":
    dope = DOPE_VGG19(num_stages=6)
    # assert dope.count_params() == 35852278
    assert dope.count_params() == 34672630  # ERROR WITH CONCAT?
    assert dope.input_shape == (None, 400, 400, 3)
    assert len(dope.output_shape) == 6
    # assert len(dope.layers) == 66
    assert len(dope.layers) == 67
    assert dope.output_shape == [(None, 50, 50, 9),
                                 (None, 50, 50, 9),
                                 (None, 50, 50, 9),
                                 (None, 50, 50, 9),
                                 (None, 50, 50, 9),
                                 (None, 50, 50, 9)]
