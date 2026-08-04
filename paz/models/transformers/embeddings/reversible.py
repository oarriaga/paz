from keras import KerasTensor
from keras import ops
from keras.layers import Embedding


class ReversibleEmbedding(Embedding):
    """Embedding that can also project hidden states back to the vocabulary.

    Called normally it looks up token embeddings. Called with ``reverse=True``
    it projects from ``output_dim`` back to ``input_dim``. With tied weights
    (the default) the reverse projection reuses the transposed embedding
    matrix; otherwise it uses a separate ``reverse_embeddings`` weight.
    """

    def __init__(self, input_dim, output_dim, tie_weights=True, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)
        self.tie_weights = tie_weights

    def build(self, inputs_shape=None):
        super().build(inputs_shape)
        if self.tie_weights:
            return
        kwargs = dict(name="reverse_embeddings", dtype=self.dtype)
        kwargs["shape"] = (self.output_dim, self.input_dim)
        kwargs["initializer"] = self.embeddings_initializer
        self.reverse_embeddings = self.add_weight(**kwargs)

    def call(self, inputs, reverse=False):
        if not reverse:
            return super().call(inputs)
        return ops.matmul(inputs, self.reverse_kernel())

    def reverse_kernel(self):
        if self.tie_weights:
            return ops.transpose(ops.convert_to_tensor(self.embeddings))
        return self.reverse_embeddings

    def compute_output_spec(self, inputs, reverse=False):
        shape = list(inputs.shape)
        if reverse:
            shape[-1] = self.input_dim
        else:
            shape.append(self.output_dim)
        return KerasTensor(shape, dtype=self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config["tie_weights"] = self.tie_weights
        return config
