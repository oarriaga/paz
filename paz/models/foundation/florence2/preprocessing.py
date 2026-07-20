"""Image preprocessing matching the FLOWER LIBERO validation transform:
uint8 bilinear-antialias resize to 112, scale by 1/255, CLIP statistics.
"""
import jax.numpy as jp

from paz.backend.image import resize


def preprocess(image):
    image = jp.asarray(image, "float32")
    if image.shape[:2] != (112, 112):
        image = resize(image, (112, 112), "linear", antialias=True)
        image = jp.clip(jp.round(image), 0.0, 255.0)
    image = image / 255.0
    mean = jp.array((0.48145466, 0.4578275, 0.40821073))
    stdv = jp.array((0.26862954, 0.26130258, 0.27577711))
    image = (image - mean) / stdv
    return image[None, ...]
