import jax
import jax.numpy as jp
import paz


def compute_mean(x):
    return jp.mean(x, axis=0)


def subtract_mean(x, mean):
    return x - mean


def compute_feature_covariance_matrix(x):
    return jp.cov(x.T)


def sort_in_descending_order(eigenvalues, eigenvectors):
    return eigenvalues[::-1], eigenvectors[::-1]


def compute_eigenvectors(covariance_matrix):
    eigenvalues, eigenvectors = jp.linalg.eigh(covariance_matrix)
    return sort_in_descending_order(eigenvalues, eigenvectors.T)


def trim_components(eigenvalues, eigenvectors, num_components):
    return eigenvalues[:num_components], eigenvectors[:num_components]


def fit(x, num_components):
    mean = compute_mean(x)
    x = subtract_mean(x, mean)

    U, values, vectors = jax.scipy.linalg.svd(x, full_matrices=False)

    # covariance_matrix = compute_feature_covariance_matrix(x)
    # values, vectors = compute_eigenvectors(covariance_matrix)
    values, vectors = trim_components(values, vectors, num_components)
    variances = values**2 / (len(x) - 1)
    state = dict(mean=mean, variances=variances, base=vectors)
    return paz.NamedTuple("PCAState", **state)


def fit_randomized(x, num_components, rng, n_iter=5):
    n_samples, n_features = x.shape
    means = compute_mean(x)
    x = x - means

    # Generate n_features normal vectors of the given size
    size = jp.minimum(2 * num_components, n_features)
    Q = jax.random.normal(rng, shape=(n_features, size))

    def step_fn(q, _):
        q, _ = jax.scipy.linalg.lu(x @ q, permute_l=True)
        q, _ = jax.scipy.linalg.lu(x.T @ q, permute_l=True)
        return q, None

    Q, _ = jax.lax.scan(step_fn, init=Q, xs=None, length=n_iter)
    Q, _ = jax.scipy.linalg.qr(x @ Q, mode="economic")
    B = Q.T @ x

    _, S, Vt = jax.scipy.linalg.svd(B, full_matrices=False)

    explained_variance = (S[:num_components] ** 2) / (n_samples - 1)
    A = Vt[:num_components]
    return paz.NamedTuple(
        "PCAState", mean=means, variances=explained_variance, base=A
    )


def transform(x, state):
    x = subtract_mean(x, state.mean)
    return (state.base @ x.T).T
