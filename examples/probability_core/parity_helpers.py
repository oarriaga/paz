import jax
import jax.numpy as jp


def assert_close(ours, theirs, atol=1e-6, rtol=1e-6):
    ours = jp.asarray(ours)
    theirs = jp.asarray(theirs)
    assert ours.shape == theirs.shape
    assert jp.allclose(ours, theirs, atol=atol, rtol=rtol, equal_nan=True)


def assert_mask_parity(ours, theirs):
    ours = jp.asarray(ours)
    theirs = jp.asarray(theirs)
    assert ours.shape == theirs.shape
    assert jp.array_equal(jp.isnan(ours), jp.isnan(theirs))
    assert jp.array_equal(jp.isposinf(ours), jp.isposinf(theirs))
    assert jp.array_equal(jp.isneginf(ours), jp.isneginf(theirs))
    assert jp.array_equal(jp.isfinite(ours), jp.isfinite(theirs))


def assert_exact_parity(ours, theirs, atol=1e-10, rtol=1e-10):
    ours = jp.asarray(ours)
    theirs = jp.asarray(theirs)
    assert_mask_parity(ours, theirs)
    finite = jp.isfinite(ours) & jp.isfinite(theirs)
    if int(finite.sum()) == 0:
        return
    assert jp.allclose(
        ours[finite], theirs[finite], atol=atol, rtol=rtol
    )


def assert_tree_exact_parity(ours, theirs, atol=1e-10, rtol=1e-10):
    our_leaves, our_tree = jax.tree_util.tree_flatten(ours)
    their_leaves, their_tree = jax.tree_util.tree_flatten(theirs)
    assert our_tree == their_tree
    assert len(our_leaves) == len(their_leaves)
    for our_leaf, their_leaf in zip(our_leaves, their_leaves):
        assert_exact_parity(our_leaf, their_leaf, atol, rtol)


def assert_distribution_parity(ours, theirs, values, atol=1e-6, rtol=1e-6):
    assert tuple(ours.batch_shape) == tuple(theirs.batch_shape)
    assert tuple(ours.event_shape) == tuple(theirs.event_shape)
    our_values = _cast_float_arg(values, _get_target_dtype(ours))
    their_values = _cast_float_arg(values, _get_target_dtype(theirs))
    assert_close(
        ours.log_prob(our_values), theirs.log_prob(their_values), atol, rtol
    )
    assert_close(ours.prob(our_values), theirs.prob(their_values), atol, rtol)


def assert_method_parity(ours, theirs, method_name, *args, atol=1e-6, rtol=1e-6):
    our_args = _cast_float_args(args, _get_target_dtype(ours))
    their_args = _cast_float_args(args, _get_target_dtype(theirs))
    our_value = getattr(ours, method_name)(*our_args)
    their_value = getattr(theirs, method_name)(*their_args)
    assert_close(our_value, their_value, atol, rtol)


def assert_exact_method_parity(
    ours, theirs, method_name, *args, atol=1e-10, rtol=1e-10
):
    our_args = _cast_float_args(args, _get_target_dtype(ours))
    their_args = _cast_float_args(args, _get_target_dtype(theirs))
    our_value = getattr(ours, method_name)(*our_args)
    their_value = getattr(theirs, method_name)(*their_args)
    assert_exact_parity(our_value, their_value, atol, rtol)


def assert_bijector_parity(
    ours, theirs, values, event_ndims=0, atol=1e-6, rtol=1e-6
):
    our_values = _cast_float_arg(values, _get_target_dtype(ours))
    their_values = _cast_float_arg(values, _get_target_dtype(theirs))
    their_forward = theirs(their_values)
    assert_close(ours(our_values), their_forward, atol, rtol)
    assert_close(ours.inverse(their_forward), theirs.inverse(their_forward))
    assert_close(
        ours.forward_log_det_jacobian(our_values, event_ndims),
        theirs.forward_log_det_jacobian(their_values, event_ndims),
        atol,
        rtol,
    )


def assert_gradient_parity(our_loss, their_loss, args, atol=1e-10, rtol=1e-10):
    argnums = _build_argnums(args)
    our_value, our_grad = jax.value_and_grad(our_loss, argnums)(*args)
    their_value, their_grad = jax.value_and_grad(their_loss, argnums)(*args)
    assert_exact_parity(our_value, their_value, atol, rtol)
    assert_tree_exact_parity(our_grad, their_grad, atol, rtol)


def assert_sample_parity(
    ours, theirs, num_samples, key, atol_mean=5e-2, atol_std=5e-2
):
    device = _get_cpu_device()
    with jax.default_device(device):
        our_samples = jp.asarray(ours.sample(num_samples, seed=key))
        their_samples = jp.asarray(theirs.sample(num_samples, seed=key))
    assert our_samples.shape == their_samples.shape
    assert_close(jp.mean(our_samples, axis=0), jp.mean(their_samples, axis=0),
                 atol_mean, 0.0)
    assert_close(jp.std(our_samples, axis=0), jp.std(their_samples, axis=0),
                 atol_std, 0.0)


def assert_method_gradient_parity(
    build_ours,
    build_theirs,
    method_name,
    params,
    method_args=(),
    reduce_fn=jp.sum,
    atol=1e-10,
    rtol=1e-10,
):
    params = _cast_float_args(params, jp.float64)
    method_args = _cast_float_args(method_args, jp.float64)

    def our_loss(*raw):
        values = getattr(build_ours(*raw), method_name)(*method_args)
        return reduce_fn(jp.asarray(values))

    def their_loss(*raw):
        values = getattr(build_theirs(*raw), method_name)(*method_args)
        return reduce_fn(jp.asarray(values))

    assert_gradient_parity(our_loss, their_loss, params, atol, rtol)


def assert_sample_gradient_parity(
    build_ours,
    build_theirs,
    params,
    seed,
    reduce_fn=None,
    atol=1e-10,
    rtol=1e-10,
):
    params = _cast_float_args(params, jp.float64)
    if reduce_fn is None:
        reduce_fn = compute_sample_square_loss

    def our_loss(*raw):
        with jax.default_device(_get_cpu_device()):
            sample = build_ours(*raw).sample(16, seed=seed)
        return reduce_fn(jp.asarray(sample))

    def their_loss(*raw):
        with jax.default_device(_get_cpu_device()):
            sample = build_theirs(*raw).sample(16, seed=seed)
        return reduce_fn(jp.asarray(sample))

    assert_gradient_parity(our_loss, their_loss, params, atol, rtol)


def compute_sample_square_loss(values):
    return jp.mean(jp.asarray(values) ** 2)


def _build_argnums(args):
    if len(args) == 1:
        return 0
    return tuple(range(len(args)))


def _cast_float_args(args, dtype):
    return tuple(_cast_float_arg(arg, dtype) for arg in args)


def _cast_float_arg(arg, dtype):
    if dtype is None:
        return arg
    values = jp.asarray(arg)
    if not jp.issubdtype(values.dtype, jp.inexact):
        return arg
    return values.astype(dtype)


def _get_target_dtype(instance):
    dtype = getattr(instance, "dtype", None)
    if _is_concrete_dtype(dtype):
        return dtype
    return _get_nested_dtype(instance)


def _is_concrete_dtype(dtype):
    if dtype is None or dtype == "?":
        return False
    if isinstance(dtype, tuple):
        return False
    return True


def _get_nested_dtype(instance):
    if hasattr(instance, "bijectors"):
        return _get_first_dtype(instance.bijectors)
    if hasattr(instance, "bijector"):
        return _get_target_dtype(instance.bijector)
    if hasattr(instance, "distribution"):
        return _get_target_dtype(instance.distribution)
    if hasattr(instance, "components_distribution"):
        return _get_target_dtype(instance.components_distribution)
    if hasattr(instance, "mixture_distribution"):
        return _get_target_dtype(instance.mixture_distribution)
    return None


def _get_first_dtype(instances):
    for instance in instances:
        dtype = _get_target_dtype(instance)
        if dtype is not None:
            return dtype
    return None


def _get_cpu_device():
    return jax.devices("cpu")[0]
