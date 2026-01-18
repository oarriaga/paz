from paz.inference.pgm.builder import (
    build_pgm_context,
    get_latent_nodes,
    get_non_root_nodes,
    get_observable_nodes,
    search_nodes,
)
from paz.inference.pgm.ops import (
    build_apply,
    build_compile,
    build_fit,
    build_infer,
    build_likelihood_log_prob,
    build_prior_log_prob,
    build_prior_prob,
    build_prior_sample,
    build_sample_forward,
    build_sample_inverse,
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
    pgm_likelihood = Likelihood(likelihood_log_prob, context.latent_space, None)
    fit = build_fit()
    pgm = Variable(
        apply,
        sample_forward,
        sample_inverse,
        name,
        [],
        None,
        context.metadata,
        pgm_prior,
        pgm_likelihood,
        None,
        None,
        None,
        fit,
        inference_defaults,
    )
    compile = build_compile(inference_defaults, lambda: pgm)
    tune = compile
    infer = build_infer(pgm_prior, pgm_likelihood, inference_defaults)
    return pgm._replace(
        compile=compile,
        tune=tune,
        infer=infer,
        fit=fit,
    )


def get_edges(nodes):
    return [[edge.name, node.name] for node in nodes for edge in node.edges]
