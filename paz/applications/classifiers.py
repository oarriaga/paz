import jax.numpy as jp
import jax
import paz


def ClassifyMiniXceptionFER():
    model = paz.models.MiniXceptionFER()
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
