from paz.models import UNET_VGG16, UNET_VGG19, UNET_RESNET50


def test_shapes_of_UNETVGG19():
    model = UNET_VGG19(weights=None)
    assert model.input_shape[1:3] == model.output_shape[1:3]


def test_shapes_of_UNETVGG16():
    model = UNET_VGG16(weights=None)
    assert model.input_shape[1:3] == model.output_shape[1:3]


def test_shapes_of_UNET_RESNET50V2():
    model = UNET_RESNET50(weights=None)
    assert model.input_shape[1:3] == model.output_shape[1:3]
