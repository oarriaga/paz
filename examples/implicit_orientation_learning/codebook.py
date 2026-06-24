from collections import namedtuple

import numpy as np

import scenes

Codebook = namedtuple("Codebook", ["views", "poses", "latents"])


def build_codebook(encoder, render_fn, poses):
    views = scenes.render_views(render_fn, poses)
    images = views.astype("float32") / 255.0
    latents = np.asarray(encoder.predict(images, verbose=0))
    return Codebook(views, np.stack([np.asarray(p) for p in poses]),
                    unit_rows(latents))


def closest_view(encoder, image, codebook):
    latent = np.asarray(encoder(image[None]))[0]
    similarities = codebook.latents @ unit_rows(latent[None])[0]
    closest = int(np.argmax(similarities))
    return codebook.views[closest], codebook.poses[closest]


def unit_rows(vectors):
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-8)
