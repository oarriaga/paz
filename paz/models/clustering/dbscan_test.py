import jax
import jax.numpy as jp
from paz.models.clustering.dbscan import dbscan


def test_dbscan_keeps_long_chain_in_one_cluster():
    x = 0.1 * jp.arange(300.0)
    points = jp.stack([x, jp.zeros(300)], axis=1)
    points = jax.random.permutation(jax.random.PRNGKey(0), points)
    labels = dbscan(points, eps=0.15, min_samples=2)
    assert jp.all(labels == 0)


def test_dbscan_separates_compact_blobs_and_noise():
    noise = 0.01 * jax.random.normal(jax.random.PRNGKey(0), (40, 2))
    centers = jp.repeat(jp.array([[0.0, 0.0], [1.0, 1.0]]), 20, axis=0)
    outlier = jp.array([[5.0, 5.0]])
    points = jp.concatenate([centers + noise, outlier])
    labels = dbscan(points, eps=0.1, min_samples=3)
    assert jp.all(labels[:20] == labels[0])
    assert jp.all(labels[20:40] == labels[20])
    assert labels[0] != labels[20]
    assert labels[40] == -1
