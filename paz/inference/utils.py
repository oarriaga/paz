import jax


def validate_space(space):
    if space not in ("inv", "fwd"):
        raise ValueError("space must be 'inv' or 'fwd'")


def get_leading_batch_size(samples):
    leaves = jax.tree_util.tree_leaves(samples)
    shaped = [leaf for leaf in leaves if hasattr(leaf, "shape")]
    if len(shaped) == 0:
        return None
    if any(len(leaf.shape) == 0 for leaf in shaped):
        return None
    first_dim = shaped[0].shape[0]
    if any(leaf.shape[0] != first_dim for leaf in shaped):
        return None
    return first_dim


def slice_batch(samples, batch_index):
    return jax.tree_util.tree_map(lambda value: value[batch_index], samples)


def squeeze_pytree(pytree):
    def squeeze_leaf(leaf):
        if hasattr(leaf, "shape") and len(leaf.shape) > 0 and leaf.shape[0] == 1:
            return jax.numpy.squeeze(leaf, axis=0)
        return leaf

    return jax.tree.map(squeeze_leaf, pytree)
