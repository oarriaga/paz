def get_inputs(pgm):
    return pgm.inputs


def get_non_priors(pgm):
    return pgm.non_priors


def get_latent_nodes(pgm):
    return pgm.latent_nodes


def get_output_nodes(pgm):
    return pgm.output_nodes


def get_observable_nodes(pgm):
    return pgm.observable_nodes


def get_nodes(pgm):
    return pgm.nodes


def get_distribution_fn(node):
    return node.distribution_fn


def get_bijector(node):
    return node.bijector
