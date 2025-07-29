import keras
from keras import ops


def compute_pairwise_distances(x, y):
    """Compute euclidean distance for each vector x with each vector y

    # Arguments:
        x: Tensor with shape `(n, vector_dim)`
        y: Tensor with shape `(m, vector_dim)`

    # Returns:
        Tensor with shape `(n, m)` where each value pair n, m corresponds to
        the distance between the vector `n` of `x` with the vector `m` of `y`
    """
    n = x.shape[0]
    m = y.shape[0]
    x = ops.tile(ops.expand_dims(x, 1), [1, m, 1])
    y = ops.tile(ops.expand_dims(y, 0), [n, 1, 1])
    return ops.mean(ops.power(x - y, 2), 2)


@keras.saving.register_keras_serializable("layers")
class ComputePairwiseDistances(keras.Layer):
    def __init__(self, **kwargs):
        super(ComputePairwiseDistances, self).__init__(**kwargs)

    def call(self, z_queries, class_prototypes):
        return compute_pairwise_distances(z_queries, class_prototypes)
