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
    ],
    defaults=[None],
)
NodeMetadata = namedtuple("NodeMetadata", ["distribution_fn", "bijector"])
PGMMetadata = namedtuple(
    "PGMMetadata", ["nodes", "inputs", "non_priors", "latent_nodes"]
)
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
