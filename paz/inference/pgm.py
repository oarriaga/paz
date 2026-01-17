from paz.inference.pgm_builder import (
    build_pgm_context,
    get_latent_nodes,
    get_non_root_nodes,
    get_observable_nodes,
    search_nodes,
)
from paz.inference.pgm_ops import (
    build_apply,
    build_data_mapping,
    build_infer,
    build_likelihood_log_prob,
    build_prior_log_prob,
    build_prior_prob,
    build_prior_sample,
    build_sample_forward,
    build_sample_inverse,
    build_tune,
    get_namedtuple_value,
)
from paz.inference.types import Density, Likelihood, Variable


def PGM(inputs, outputs, name):
    context = build_pgm_context(inputs, outputs, name)
    sample_inverse = build_sample_inverse(context)
    sample_forward = build_sample_forward(context)
    apply = build_apply(context)
    prior_sample = build_prior_sample(context, sample_inverse)
    prior_log_prob = build_prior_log_prob(context)
    prior_prob = build_prior_prob(prior_log_prob)
    likelihood_log_prob = build_likelihood_log_prob(context, apply)
    inference_defaults = {}
    pgm_prior = Density(
        prior_sample, prior_log_prob, prior_prob, context.latent_space, None
    )
    pgm_likelihood = Likelihood(
        likelihood_log_prob, context.latent_space, None
    )
    tune = build_tune(pgm_prior, pgm_likelihood, inference_defaults)
    infer = build_infer(pgm_prior, pgm_likelihood, inference_defaults)
    return Variable(
        apply,
        sample_forward,
        sample_inverse,
        name,
        [],
        None,
        context.metadata,
        pgm_prior,
        pgm_likelihood,
        tune,
        infer,
        inference_defaults,
    )


def get_edges(nodes):
    return [
        [edge.name, node.name] for node in nodes for edge in node.edges
    ]
