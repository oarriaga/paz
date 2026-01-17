def get_pgm_metadata(pgm):
    metadata = getattr(pgm, "metadata", None)
    if metadata is None:
        raise ValueError("Expected a PGM Variable with metadata.")
    return metadata


def get_node_metadata(node):
    metadata = getattr(node, "metadata", None)
    if metadata is None:
        raise ValueError("Expected a Variable with metadata.")
    return metadata


def get_inputs(pgm):
    return get_pgm_metadata(pgm).inputs


def get_non_priors(pgm):
    return get_pgm_metadata(pgm).non_priors


def get_latent_nodes(pgm):
    return get_pgm_metadata(pgm).latent_nodes


def get_nodes(pgm):
    return get_pgm_metadata(pgm).nodes


def get_distribution_fn(node):
    return get_node_metadata(node).distribution_fn


def get_bijector(node):
    return get_node_metadata(node).bijector
