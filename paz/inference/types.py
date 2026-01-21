from collections import namedtuple
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

Distribution = tfd.Distribution
NodeState = namedtuple("NodeState", ["sample", "log_prob", "log_prob_sum"])
Variable = namedtuple(
    "Variable",
    [
        "log_prob",
        "log_prob_inverse",
        "sample",
        "sample_inverse",
        "name",
        "edges",
        "distribution",
        "distribution_fn",
        "bijector",
        "fit",
        "fit_bijector",
    ],
    defaults=[None] * 11,
)
PGM = namedtuple(
    "PGM",
    [
        "sample",
        "sample_inverse",
        "name",
        "nodes",
        "inputs",
        "non_priors",
        "latent_nodes",
        "output_nodes",
        "observable_nodes",
        "latent_space",
        "prior",
        "likelihood",
        "configure",
        "tune",
        "infer",
        "fit",
        "inference_defaults",
    ],
    defaults=[None] * 17,
)
Density = namedtuple(
    "Density",
    [
        "sample",
        "sample_inverse",
        "log_prob",
        "log_prob_inverse",
        "prob",
        "prob_inverse",
        "latent_space",
        "metadata",
    ],
)
Likelihood = namedtuple(
    "Likelihood",
    ["log_prob", "log_prob_inverse", "latent_space", "metadata"],
)
DiscretePosterior = namedtuple(
    "DiscretePosterior",
    ["support", "log_posterior", "posterior", "z_map", "z_map_value"],
)
MCMCPosteriorType = namedtuple(
    "MCMCPosterior",
    [
        "sample",
        "sample_inverse",
        "diagnostics",
        "as_density",
        "to_empirical",
        "update",
        "samples",
        "inverse_samples",
        "infos",
        "config",
        "latent_space",
    ],
)


def SampleType(node_names):
    if isinstance(node_names, list):
        Sample = namedtuple("Sample", [name for name in node_names])
    else:
        raise ValueError("'node_names' must be a list of strings")
    return Sample
