from paz.models.classification.xception import build_xception


def XCEPTION_MINI(input_shape, num_classes):
    stem_kernels, block_data = [32, 64], [128, 128, 256, 256, 512, 512, 1024]
    model_inputs = (input_shape, num_classes, stem_kernels, block_data)
    model = build_xception(*model_inputs)
    model._name = 'XCEPTION-MINI'
    return model
