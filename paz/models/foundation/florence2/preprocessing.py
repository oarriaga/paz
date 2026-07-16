"""Image preprocessing matching the FLOWER LIBERO validation transform:
uint8 bilinear-antialias resize to 112, scale by 1/255, CLIP statistics.
"""
import jax.numpy as jp

from paz.backend.image import resize

IMAGE_SIZE = 112
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STDV = (0.26862954, 0.26130258, 0.27577711)


def preprocess(image):
    image = jp.asarray(image, "float32")
    size = (IMAGE_SIZE, IMAGE_SIZE)
    if image.shape[:2] != size:
        image = resize(image, size, "linear", antialias=True)
        image = jp.clip(jp.round(image), 0.0, 255.0)
    image = image / 255.0
    image = (image - jp.array(CLIP_MEAN)) / jp.array(CLIP_STDV)
    return image[None, ...]
