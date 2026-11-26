import jax.numpy as jp
import jax
import paz


def ClassifyMiniXceptionFER():
    """Builds a MiniXception classifier with FER weights.

    # Returns
    Callable taking an RGB image. It resizes the image to the model input
    size, converts it to grayscale and returns the class scores with the
    batch axis removed.
    """
    return ClassifyMiniXception(paz.models.MiniXceptionFER())


def ClassifyMiniXceptionIMDB():
    """Builds a MiniXception classifier with IMDB weights.

    # Returns
    Callable taking an RGB image. It resizes the image to the model input
    size, converts it to grayscale and returns the class scores with the
    batch axis removed.
    """
    return ClassifyMiniXception(paz.models.MiniXceptionIMDB())


def ClassifyMiniXception(model):
    """Builds an image classifier from a MiniXception `model`.

    # Arguments
    - `model`: MiniXception Keras model.

    # Returns
    Callable taking an RGB image. It resizes the image to the model input
    size, converts it to grayscale and returns the class scores with the
    batch axis removed.
    """
    resize = paz.lock(paz.image.resize_opencv, paz.image.get_input_size(model))

    def preprocess(image):
        image = paz.image.normalize(image)
        image = paz.image.rgb_to_gray(image)
        return jp.expand_dims(image, [0, -1])

    def postprocess(scores):
        return jp.squeeze(scores, axis=0)

    @jax.jit
    def call(image):
        return postprocess(model(preprocess(image)))

    return lambda image: call(resize(image))  # split static vs dynamic for jit
