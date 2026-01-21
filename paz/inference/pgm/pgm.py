from paz.inference.pgm.builder import (
    build_pgm_context,
    get_latent_nodes,
    get_non_root_nodes,
    get_observable_nodes,
    search_nodes,
)
from paz.inference.pgm.ops import (
    build_configure,
    build_fit,
    build_infer,
    build_likelihood_log_prob,
    build_prior_log_prob,
    build_prior_prob,
    build_prior_sample,
    build_sample_forward,
    build_sample_inverse,
    build_sample_predictive,
)
from paz.inference.types import Density, Likelihood, PGM as PGMType


def PGM(inputs, outputs, name):
    context = build_pgm_context(inputs, outputs, name)
    sample_inverse = build_sample_inverse(context)
    sample_forward = build_sample_forward(context)
    sample_predictive = build_sample_predictive(context)
    prior_sample, prior_sample_inverse = build_prior_sample(
        context, sample_inverse
    )
    prior_log_prob, prior_log_prob_inverse = build_prior_log_prob(context)
    prior_prob, prior_prob_inverse = build_prior_prob(
        prior_log_prob, prior_log_prob_inverse
    )
    likelihood_log_prob, likelihood_log_prob_inverse = build_likelihood_log_prob(
        context
    )
    inference_defaults = {}
    pgm_prior = Density(
        prior_sample,
        prior_sample_inverse,
        prior_log_prob,
        prior_log_prob_inverse,
        prior_prob,
        prior_prob_inverse,
        context.latent_space,
        None,
    )
    pgm_likelihood = Likelihood(
        likelihood_log_prob,
        likelihood_log_prob_inverse,
        context.latent_space,
        None,
    )
    fit = build_fit()
    configure = build_configure(inference_defaults, lambda: pgm)
    tune = configure
    infer = build_infer(
        pgm_prior,
        pgm_likelihood,
        inference_defaults,
        sample_predictive=sample_predictive,
    )
    pgm = PGMType(
        sample_forward,
        sample_inverse,
        name,
        context.nodes,
        context.inputs,
        context.non_priors,
        context.latent_nodes,
        context.output_nodes,
        context.observable_nodes,
        context.latent_space,
        pgm_prior,
        pgm_likelihood,
        configure,
        tune,
        infer,
        fit,
        inference_defaults,
    )
    return pgm


def get_edges(nodes):
    return [[edge.name, node.name] for node in nodes for edge in node.edges]
