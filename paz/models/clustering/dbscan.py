from functools import partial
import jax.numpy as jnp
from jax import jit


@jit
def compute_pairwise_distances_squared(points):
    """Compute squared Euclidean distance between all pairs of points."""
    norms_squared = jnp.sum(points**2, axis=1)
    dot_products = points @ points.T
    distances_squared = (
        norms_squared[:, None] + norms_squared[None, :] - 2 * dot_products
    )
    return jnp.maximum(distances_squared, 0.0)  # Numerical stability


@jit
def compute_pairwise_distances(points):
    """Compute Euclidean distance between all pairs of points."""
    return jnp.sqrt(compute_pairwise_distances_squared(points))


@partial(jit, static_argnames=["eps"])
def find_adjacency_matrix(points, eps):
    """Find which points are neighbors (within eps distance)."""
    distances_squared = compute_pairwise_distances_squared(points)
    eps_squared = eps**2
    return distances_squared <= eps_squared


@partial(jit, static_argnames=["eps"])
def count_neighbors(points, eps):
    """Count how many neighbors each point has within eps distance."""
    adjacency = find_adjacency_matrix(points, eps)
    return jnp.sum(adjacency, axis=1)


@partial(jit, static_argnames=["eps", "min_samples"])
def identify_core_points(points, eps, min_samples):
    """Identify core points (points with at least min_samples neighbors)."""
    neighbor_counts = count_neighbors(points, eps)
    return neighbor_counts >= min_samples


@jit
def propagate_labels_one_step(labels, adjacency):
    """Propagate minimum label to all neighbors."""
    # For each point, find minimum label among neighbors
    # Use large value for non-neighbors
    large_value = labels.shape[0] + 1
    neighbor_labels = jnp.where(adjacency, labels[None, :], large_value)
    min_neighbor_label = jnp.min(neighbor_labels, axis=1)
    return jnp.minimum(labels, min_neighbor_label)


def find_connected_components(adjacency, max_iterations=100):
    """Find connected components using iterative label propagation."""
    n_points = adjacency.shape[0]

    # Initialize: each point is its own label
    labels = jnp.arange(n_points)

    # Iterate until convergence
    for _ in range(max_iterations):
        new_labels = propagate_labels_one_step(labels, adjacency)
        if jnp.all(new_labels == labels):
            break
        labels = new_labels

    # Relabel to consecutive integers (0, 1, 2, ...)
    unique_labels = jnp.unique(labels)
    label_map = jnp.zeros(n_points + 1, dtype=jnp.int32)
    label_map = label_map.at[unique_labels].set(jnp.arange(len(unique_labels)))

    return label_map[labels]


def assign_border_points(points, labels, adjacency, is_core):
    """Assign non-core points to the cluster of their nearest core neighbor."""
    # n_points = points.shape[0]
    is_border = ~is_core & (labels == -1)

    # For each border point, find if it has any core neighbors
    has_core_neighbor = adjacency & is_core[None, :]  # (N, N)

    # For border points, find the nearest core neighbor
    distances = compute_pairwise_distances(points)

    # Set distance to non-core-neighbors to infinity
    large_value = 1e10
    distances_to_core = jnp.where(has_core_neighbor, distances, large_value)

    # Find nearest core neighbor for each point
    nearest_core_idx = jnp.argmin(distances_to_core, axis=1)
    nearest_core_label = labels[nearest_core_idx]

    # Check if actually has a core neighbor within eps
    has_any_core_neighbor = jnp.any(has_core_neighbor, axis=1)

    # Update labels for border points
    should_update = is_border & has_any_core_neighbor
    new_labels = jnp.where(should_update, nearest_core_label, labels)

    return new_labels


def dbscan(points, eps, min_samples):
    """DBSCAN clustering algorithm."""
    n_points = points.shape[0]

    adjacency = find_adjacency_matrix(points, eps)
    is_core = identify_core_points(points, eps, min_samples)
    core_adjacency = adjacency & is_core[:, None] & is_core[None, :]
    labels = jnp.full(n_points, -1, dtype=jnp.int32)
    core_indices = jnp.where(is_core)[0]
    n_core = len(core_indices)

    if n_core == 0:
        return labels

    core_adjacency_sub = core_adjacency[core_indices][:, core_indices]
    core_labels = find_connected_components(core_adjacency_sub)
    labels = labels.at[core_indices].set(core_labels)
    labels = assign_border_points(points, labels, adjacency, is_core)

    return labels
