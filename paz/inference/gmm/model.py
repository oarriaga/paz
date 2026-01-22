import numpy as np
import jax.numpy as jp
from tensorflow_probability.substrates import jax as tfp

from paz.inference.gmm.em import fit_gmm_em
from paz.inference.observable import Observable
from paz.inference.pgm import PGM
from paz.inference.pgm.ops import build_data_mapping
from paz.inference.prior import Prior


tfd = tfp.distributions


def GMM(weights, means=None, covariances=None, covariance="diag", name="gmm"):
    weights_value, means_value, covariances_value = _prepare_parameters(
        weights, means, covariances, covariance
    )
    weights_dist = tfd.Deterministic(weights_value)
    means_dist = tfd.Deterministic(means_value)
    covariances_dist = tfd.Deterministic(covariances_value)
    weights = Prior(weights_dist, name="weights")
    means = Prior(means_dist, name="means")
    covariances = Prior(covariances_dist, name="covariances")

    def y_distribution(weights, means, covariances):
        components = _build_components(means, covariances, covariance)
        return tfd.MixtureSameFamily(tfd.Categorical(probs=weights), components)

    y = Observable(y_distribution, name="y")(weights, means, covariances)
    model = PGM([weights, means, covariances], [y], name)
    output_nodes = model.output_nodes
    num_components = weights_value.shape[0]

    def fit(key, data, method="em", **method_kwargs):
        method_name = method.lower() if method is not None else "em"
        if method_name != "em":
            raise ValueError("Unknown fit method: " + method_name)
        data_mapping = build_data_mapping(data, output_nodes)
        samples = _normalize_samples(data_mapping[output_nodes[0].name])
        fit_kwargs = dict(covariance=covariance, **method_kwargs)
        fitted = fit_gmm_em(key, samples, num_components, **fit_kwargs)
        inputs = {
            "weights": fitted.weights,
            "means": fitted.means,
            "covariances": fitted.covariances,
        }
        return _build_output_distributions(output_nodes, inputs)

    return model._replace(fit=fit)


def _build_components(means, covariances, covariance):
    if covariance == "diag":
        parameters = dict(loc=means, scale_diag=jp.sqrt(covariances))
        return tfd.MultivariateNormalDiag(**parameters)
    if covariance == "full":
        parameters = dict(loc=means, covariance_matrix=covariances)
        return tfd.MultivariateNormalFullCovariance(**parameters)
    raise ValueError("covariance must be 'diag' or 'full'")


def _normalize_samples(samples):
    samples = jp.asarray(samples)
    if samples.ndim == 1:
        return samples[:, None]
    return samples


def _build_output_distributions(output_nodes, input_values):
    output_distributions = {}
    for node in output_nodes:
        if node.distribution_fn is None:
            raise ValueError("Output node missing distribution function.")
        node_inputs = [input_values[edge.name] for edge in node.edges]
        output_distributions[node.name] = node.distribution_fn(*node_inputs)
    if len(output_distributions) == 1:
        return next(iter(output_distributions.values()))
    return output_distributions


def _prepare_parameters(weights, means, covariances, covariance):
    if means is None and covariances is None and isinstance(
        weights, (int, np.integer)
    ):
        return _default_parameters(int(weights), 1, covariance)
    num_components = _infer_num_components(weights, means, covariances)
    num_dims = _infer_num_dims(means, covariances, covariance)
    default_weights, default_means, default_covariances = _default_parameters(
        num_components, num_dims, covariance
    )
    weights = default_weights if weights is None else jp.asarray(weights)
    means = default_means if means is None else _normalize_means(means)
    covariances = (
        default_covariances
        if covariances is None
        else _normalize_covariances(covariances, covariance)
    )
    return weights, means, covariances


def _infer_num_components(weights, means, covariances):
    for value in (weights, means, covariances):
        if value is not None:
            return int(jp.asarray(value).shape[0])
    raise ValueError("num_components is required when parameters are None")


def _infer_num_dims(means, covariances, covariance):
    if means is not None:
        means = jp.asarray(means)
        return int(1 if means.ndim == 1 else means.shape[1])
    if covariances is not None:
        covariances = jp.asarray(covariances)
        if covariance == "diag":
            return int(1 if covariances.ndim == 1 else covariances.shape[1])
        return int(covariances.shape[1])
    return 1


def _normalize_means(means):
    means = jp.asarray(means)
    return means[:, None] if means.ndim == 1 else means


def _normalize_covariances(covariances, covariance):
    covariances = jp.asarray(covariances)
    if covariance == "diag" and covariances.ndim == 1:
        return covariances[:, None]
    return covariances


def _default_parameters(num_components, num_dims, covariance):
    weights = jp.full((num_components,), 1.0 / num_components)
    means = jp.zeros((num_components, num_dims))
    if covariance == "diag":
        covariances = jp.ones((num_components, num_dims))
    else:
        covariances = jp.tile(jp.eye(num_dims), (num_components, 1, 1))
    return weights, means, covariances
