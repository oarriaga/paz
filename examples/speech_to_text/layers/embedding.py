from keras import ops
from keras.layers import Embedding, Lambda

from examples.speech_to_text.layers.utils import Kernel


def embed_position(x, seq_length, trainable, positions, name):
    dim = x.shape[-1]
    embedding = build_embedding(seq_length, dim, trainable, name)
    if positions is None:
        index_name = f"{name}_indices"
        positions = build_index_lambda(index_name)(x)
        embeddings = embedding(positions)
        embeddings = ops.expand_dims(embeddings, axis=0)
        return broadcast(embeddings, x)
    if len(positions.shape) == 1:
        positions = ops.expand_dims(positions, axis=0)
    return broadcast(embedding(positions), x)


def build_embedding(seq_length, dim, trainable, name):
    kwargs = {"trainable": trainable, "name": name}
    return Embedding(seq_length, dim, Kernel(), **kwargs)


def build_index_lambda(name):
    fn = build_position_indices
    shape_fn = position_indices_shape
    return Lambda(fn, output_shape=shape_fn, name=name)


def build_position_indices(x):
    return build_position_args(ops.shape(x)[-2], 0)


def position_indices_shape(shape):
    return (shape[-2],)


def broadcast(embeddings, inputs):
    return ops.ones_like(inputs) * embeddings


def build_position_args(sequence_length, start):
    end = start + sequence_length
    return ops.arange(start, end, dtype="int32")


