from collections import namedtuple
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

Distribution = tfd.Distribution
NodeState = namedtuple("NodeState", ["sample", "log_prob", "log_prob_sum"])
Variable = namedtuple(
    "Variable",
    [
        "apply",
        "sample",
        "sample_inverse",
        "name",
        "edges",
        "distribution",
        "metadata",
        "prior",
        "likelihood",
        "compile",
        "tune",
        "infer",
        "inference_defaults",
    ],
    defaults=[None, None, None, None, None, None],
)
NodeMetadata = namedtuple("NodeMetadata", ["distribution_fn", "bijector"])
PGMMetadata = namedtuple(
    "PGMMetadata",
    [
        "nodes",
        "inputs",
        "non_priors",
        "latent_nodes",
        "output_nodes",
        "observable_nodes",
    ],
)
Density = namedtuple(
    "Density", ["sample", "log_prob", "prob", "latent_space", "metadata"]
)
Likelihood = namedtuple("Likelihood", ["log_prob", "latent_space", "metadata"])
DiscretePosterior = namedtuple(
    "DiscretePosterior",
    ["support", "log_posterior", "posterior", "z_map", "z_map_value"],
)


def SampleType(node_names):
    if isinstance(node_names, list):
        Sample = namedtuple("Sample", [name for name in node_names])
    else:
        raise ValueError("'node_names' must be a list of strings")
    return Sample
