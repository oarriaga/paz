import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import jax.numpy as jp

import paz


def test_autoencoder_reconstruction_shape():
    model = paz.models.AutoEncoder((128, 128, 3), 128)
    reconstruction = model(jp.zeros((2, 128, 128, 3)))
    assert tuple(reconstruction.shape) == (2, 128, 128, 3)
    assert float(reconstruction.min()) >= 0.0
    assert float(reconstruction.max()) <= 1.0


def test_extract_encoder_latent_shape():
    model = paz.models.AutoEncoder((128, 128, 3), 64)
    encoder = paz.models.extract_encoder(model)
    latent = encoder(jp.zeros((2, 128, 128, 3)))
    assert tuple(latent.shape) == (2, 64)
