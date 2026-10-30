import jax
import jax.numpy as jp

import paz


def build_two_blobs(key, num_points=200, stdv=0.01):
    noise = stdv * jax.random.normal(key, (2 * num_points, 3))
    left = jp.tile(jp.array([[0.0, 0.0, 1.0]]), (num_points, 1))
    right = jp.tile(jp.array([[1.0, 0.0, 1.0]]), (num_points, 1))
    return jp.concatenate([left, right]) + noise


def fit_all_valid(key, points, num_clusters):
    is_valid = jp.ones(len(points), bool)
    return paz.KMeans(key, points, is_valid, num_clusters)


def test_kmeans_finds_two_blobs():
    points = build_two_blobs(jax.random.PRNGKey(0))
    clusters = fit_all_valid(jax.random.PRNGKey(1), points, 2)
    order = jp.argsort(clusters.means[:, 0])
    expected = jp.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    assert jp.allclose(clusters.means[order], expected, atol=5e-3)
    assert jp.all(clusters.counts == 200)


def test_kmeans_recovers_blob_spread():
    points = build_two_blobs(jax.random.PRNGKey(0), num_points=2000)
    clusters = fit_all_valid(jax.random.PRNGKey(1), points, 2)
    assert jp.allclose(clusters.stdvs, 0.01, atol=1e-3)


def test_kmeans_ignores_invalid_points():
    points = build_two_blobs(jax.random.PRNGKey(0))
    far_away = jp.full((300, 3), 100.0)
    padded_points = jp.concatenate([points, far_away])
    is_valid = jp.arange(len(padded_points)) < len(points)
    args = (jax.random.PRNGKey(1), padded_points, is_valid, 2)
    clusters = paz.KMeans(*args)
    order = jp.argsort(clusters.means[:, 0])
    expected = jp.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    assert jp.allclose(clusters.means[order], expected, atol=5e-3)
    assert jp.all(clusters.counts == 200)


def test_kmeans_small_cluster_falls_back_to_all_points():
    points = build_two_blobs(jax.random.PRNGKey(0), num_points=100)
    is_valid = jp.ones(len(points), bool)
    clusters = paz.KMeans(jax.random.PRNGKey(2), points, is_valid, 150)
    is_small = clusters.counts < 3
    expected_mean = points.mean(axis=0)
    assert bool(is_small.any())
    assert jp.allclose(clusters.means[is_small], expected_mean, atol=1e-5)


def test_kmeans_is_importable_from_package():
    from paz.models.clustering import Clusters, KMeans

    assert KMeans is paz.KMeans
    assert Clusters._fields == ("means", "stdvs", "counts")
