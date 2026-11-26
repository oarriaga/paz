from paz.inference.pgm.builder import build_graph
from paz.inference.pgm.ops import (
    build_configure,
    build_fit,
    build_infer,
    build_likelihood_log_prob,
    build_log_density,
    build_prior_log_prob,
    build_prior_prob,
    build_prior_sample,
    build_sample_forward,
    build_sample_inverse,
    build_sample_predictive,
    compute_sample_shapes,
)
from paz.inference.types import Density, Likelihood, PGM as PGMType


def PGM(inputs, outputs, name):
    """Builds a probabilistic graphical model from its variables.

    # Arguments
    - `inputs`: List of `Prior` variables.
    - `outputs`: List of output variables, e.g. `Observable` variables.
    - `name`: String. Name of the model.

    # Returns
    Namedtuple with the fields `sample`, `sample_inverse`, `prior`,
    `likelihood`, `log_prob`, `log_prob_inverse`, `configure`, `infer`
    and `fit`, among others.

    # Example
    ```python
    mean = paz.Prior(paz.distributions.Normal(0.0, 1.0), name="mean")

    def likelihood(mean):
        return paz.distributions.Normal(mean, 0.5)

    y = paz.Observable(likelihood, name="y")(mean)
    model = paz.PGM([mean], [y], "gaussian")
    samples = model.sample(jax.random.PRNGKey(0), num_samples=4)
    samples.mean.shape, samples.y.shape  # (4,), (4,)
    ```
    """
    graph = build_graph(inputs, outputs, name)
    prior_sample, prior_sample_inverse = build_prior_sample(graph)
    sample_inverse = build_sample_inverse(graph, prior_sample_inverse)
    sample_forward = build_sample_forward(graph)
    sample_predictive = build_sample_predictive(graph)
    shapes = compute_sample_shapes(prior_sample)
    inverse_shapes = compute_sample_shapes(prior_sample_inverse)
    prior_log_probs = build_prior_log_prob(graph, shapes, inverse_shapes)
    prior_log_prob, prior_log_prob_inverse = prior_log_probs
    prior_prob, prior_prob_inverse = build_prior_prob(
        prior_log_prob, prior_log_prob_inverse
    )
    likelihood = build_likelihood_log_prob(graph, shapes, inverse_shapes)
    likelihood_log_prob, likelihood_log_prob_inverse = likelihood
    inference_defaults = {}
    pgm_prior = Density(
        prior_sample,
        prior_sample_inverse,
        prior_log_prob,
        prior_log_prob_inverse,
        prior_prob,
        prior_prob_inverse,
        graph.latent_space,
        None,
    )
    pgm_likelihood = Likelihood(
        likelihood_log_prob,
        likelihood_log_prob_inverse,
        graph.latent_space,
        None,
    )
    log_prob, log_prob_inverse = build_log_density(pgm_prior, pgm_likelihood)
    fit = build_fit()
    configure = build_configure(inference_defaults)
    infer = build_infer(lambda: pgm, inference_defaults, sample_predictive)
    pgm = PGMType(
        sample_forward,
        sample_inverse,
        name,
        graph.nodes,
        graph.inputs,
        graph.non_priors,
        graph.latent_nodes,
        graph.output_nodes,
        graph.observable_nodes,
        graph.latent_space,
        pgm_prior,
        pgm_likelihood,
        configure,
        infer,
        fit,
        inference_defaults,
        log_prob,
        log_prob_inverse,
    )
    return pgm
