from collections import namedtuple

import jax.numpy as jp

import scenes

Codebook = namedtuple("Codebook", ["views", "poses", "latents"])


def build_codebook(encoder, render_fn, poses):
    views = scenes.render_views(render_fn, poses)
    images = jp.asarray(views, jp.float32) / 255.0
    latents = unit_rows(jp.asarray(encoder.predict(images, verbose=0)))
    matrices = jp.stack([jp.asarray(pose) for pose in poses])
    return Codebook(jp.asarray(views), matrices, latents)


def closest_view(encoder, image, codebook):
    latent = unit_rows(jp.asarray(encoder(image[jp.newaxis])))[0]
    similarities = codebook.latents @ latent
    closest = jp.argmax(similarities)
    return codebook.views[closest], codebook.poses[closest]


def unit_rows(vectors):
    norms = jp.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / jp.maximum(norms, 1e-8)
