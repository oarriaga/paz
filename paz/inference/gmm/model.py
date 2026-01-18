import jax.numpy as jp
from tensorflow_probability.substrates import jax as tfp

from paz.inference.gmm.em import fit_gmm_em
from paz.inference.observable import Observable
from paz.inference.pgm import PGM
from paz.inference.pgm.ops import build_data_mapping
from paz.inference.prior import Prior


tfd = tfp.distributions


def GMM(weights, means, covariances, covariance="diag", name="gmm"):
    weights_value, means_value, covariances_value = weights, means, covariances
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
    output_nodes = model.metadata.output_nodes
    num_components = weights_value.shape[0]

    def fit(key, data, method="em", **method_kwargs):
        method_name = method.lower() if method is not None else "em"
        if method_name != "em":
            raise ValueError("Unknown fit method: " + method_name)
        data_mapping = build_data_mapping(data, output_nodes)
        samples = _normalize_samples(data_mapping[output_nodes[0].name])
        fit_kwargs = dict(covariance=covariance, **method_kwargs)
        fitted = fit_gmm_em(key, samples, num_components, **fit_kwargs)
        return GMM(*fitted, covariance=covariance, name=name)

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
