import os

import jax.numpy as jp
from tensorflow_probability.substrates import jax as tfp

import paz
from paz.inference.prior import Prior
from paz.inference.serialize.api import _infer_format

tfd = tfp.distributions


def test_save_load_prior_roundtrip(tmp_path):
    prior = Prior(tfd.Normal(0.0, 1.0), name="x")
    path = tmp_path / "prior"
    paz.inference.save(prior, path)
    loaded = paz.inference.load(path)
    value = jp.array(0.1)
    log_prob = prior.log_prob(value).log_prob_sum
    log_prob_loaded = loaded.log_prob(value).log_prob_sum
    assert jp.allclose(log_prob, log_prob_loaded)


def test_infer_format_from_manifest(tmp_path):
    path = tmp_path / "format"
    os.makedirs(path)
    (path / "manifest.json").write_text("{}")
    assert _infer_format(path) == "paz"


def test_infer_format_without_manifest(tmp_path):
    path = tmp_path / "empty"
    os.makedirs(path)
    try:
        _infer_format(path)
    except ValueError as error:
        assert "Unable to infer format" in str(error)
    else:
        raise AssertionError("Expected ValueError")
