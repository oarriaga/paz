from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jp
import paz

# Points marked invalid (for example padding) belong to no cluster.
Clusters = namedtuple("Clusters", ["means", "stdvs", "counts"])
compute_squared_distances = paz.pointcloud.compute_pairwise_squared_distances


@partial(jax.jit, static_argnames=["num_clusters", "num_iterations"])
def KMeans(key, points, is_valid, num_clusters, num_iterations=20):
    """Groups 3D points into num_clusters clusters with k-means.

    Starts from k-means++ centers, then moves each center to the mean of
    its points num_iterations times.

    # Arguments
        key: JAX random key for the k-means++ start.
        points: Array (num_points, 3).
        is_valid: Boolean array (num_points,); False points are ignored,
            so padded arrays keep one compiled version for every input.
        num_clusters: Int, number of clusters.
        num_iterations: Int, number of center updates.

    # Returns
        Clusters with per-cluster means, standard deviations and counts.
        Clusters with fewer than 3 points take the statistics of all
        valid points.
    """
    weights = is_valid.astype(points.dtype)
    centers = sample_initial_centers(key, points, weights, num_clusters)
    centers = move_centers_to_means(points, weights, centers, num_iterations)
    return summarize_clusters(points, weights, centers)


def sample_initial_centers(key, points, weights, num_clusters):
    # k-means++: each new center is drawn far from the centers so far
    num_points = len(points)
    key, first_key = jax.random.split(key)
    first_probabilities = weights / jp.sum(weights)
    first_arg = jax.random.choice(first_key, num_points, p=first_probabilities)
    centers = jp.zeros((num_clusters, 3)).at[0].set(points[first_arg])

    def add_center(center_arg, carry):
        centers, key = carry
        key, choice_key = jax.random.split(key)
        distances = compute_squared_distances(points, centers)
        is_sampled = jp.arange(num_clusters) < center_arg
        distances = jp.where(is_sampled, distances, jp.inf)
        nearest = weights * jp.min(distances, axis=1)
        probabilities = nearest / jp.sum(nearest)
        point_arg = jax.random.choice(choice_key, num_points, p=probabilities)
        return centers.at[center_arg].set(points[point_arg]), key

    carry = jax.lax.fori_loop(1, num_clusters, add_center, (centers, key))
    return carry[0]


def move_centers_to_means(points, weights, centers, num_iterations):
    def update(_, centers):
        memberships = assign_to_nearest(points, weights, centers)
        counts = memberships.sum(axis=0)
        means = average_per_cluster(points, memberships, counts)
        is_empty = jp.expand_dims(counts == 0, axis=1)
        return jp.where(is_empty, centers, means)

    return jax.lax.fori_loop(0, num_iterations, update, centers)


def assign_to_nearest(points, weights, centers):
    # one-hot rows: which center each point is closest to (none if padded)
    distances = compute_squared_distances(points, centers)
    labels = jp.argmin(distances, axis=1)
    one_hot = jax.nn.one_hot(labels, len(centers))
    return one_hot * jp.expand_dims(weights, axis=1)


def average_per_cluster(values, memberships, counts):
    sums = memberships.T @ values
    return sums / jp.expand_dims(jp.maximum(counts, 1), axis=1)


def summarize_clusters(points, weights, centers, min_count=3):
    memberships = assign_to_nearest(points, weights, centers)
    counts = memberships.sum(axis=0)
    means = average_per_cluster(points, memberships, counts)
    residuals = points - (memberships @ means)
    variances = average_per_cluster(residuals ** 2, memberships, counts)
    stdvs = jp.maximum(jp.sqrt(variances), 1e-6)
    # clusters with almost no points fall back to the whole object
    object_mean, object_stdv = compute_object_statistics(points, weights)
    is_small = jp.expand_dims(counts < min_count, axis=1)
    means = jp.where(is_small, object_mean, means)
    stdvs = jp.where(is_small, object_stdv, stdvs)
    return Clusters(means, stdvs, counts.astype(jp.int32))


def compute_object_statistics(points, weights):
    # mean and spread of all valid points
    column = jp.expand_dims(weights, axis=1)
    count = jp.sum(weights)
    mean = jp.sum(column * points, axis=0) / count
    variance = jp.sum(column * ((points - mean) ** 2), axis=0) / count
    return mean, jp.maximum(jp.sqrt(variance), 1e-6)
