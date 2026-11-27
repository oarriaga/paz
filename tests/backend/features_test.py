import jax.numpy as jp

import paz


def test_mutual_nearest_neighbors_applies_negative_min_cosine():
    descriptors_a = jp.array([[1.0, 0.0]])
    descriptors_b = jp.array([[-0.7, 0.714]])
    match = paz.features.find_mutual_nearest_neighbors
    source, target = match(descriptors_a, descriptors_b, min_cosine=-0.5)
    assert source.shape == (0,)
    assert target.shape == (0,)
    source, target = match(descriptors_a, descriptors_b, min_cosine=-0.9)
    assert source.tolist() == [0]
    assert target.tolist() == [0]
    source, target = match(descriptors_a, descriptors_b)
    assert source.tolist() == [0]
