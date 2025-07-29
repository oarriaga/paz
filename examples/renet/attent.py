from keras import ops


def gaussian_normalize(x, axis, epsilon=1e-5):
    """Normalizes input to have zero mean and unit variance across an axis."""
    mean = ops.mean(x, axis=axis, keepdims=True)
    variance = ops.var(x, axis=axis, keepdims=True)
    return (x - mean) / ops.sqrt(variance + epsilon)


def attent(axis, sum_axes, correlations, shape, temperature):
    (num_queries, num_ways, Hs, Ws, Hq, Wq) = shape
    _shape = (num_queries, num_ways, Hs * Ws, Hq, Wq)
    correlations = gaussian_normalize(ops.reshape(correlations, _shape), axis)
    correlations = ops.softmax(correlations / temperature, axis)
    attention = ops.sum(ops.reshape(correlations, shape), axis=sum_axes)
    return ops.expand_dims(attention, axis=2)
