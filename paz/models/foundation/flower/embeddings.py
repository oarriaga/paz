from keras import ops
from keras.layers import Dense

from paz.models.transformers.embeddings import timestep


def embed_flow_time(times, hidden_dim):
    features = timestep.sinusoidal(times, 256, 10_000.0, 1000.0)
    return embed_features(features, hidden_dim, "flow_time_embedder")


def embed_frequency(frequencies, hidden_dim):
    features = timestep.sinusoidal(frequencies, 256, 1000.0, 1.0)
    return embed_features(features, hidden_dim, "frequency_embedder")


def embed_features(features, hidden_dim, name):
    inner = Dense(hidden_dim, activation="silu", name=f"{name}_dense_1")
    outer = Dense(hidden_dim, name=f"{name}_dense_2")
    return outer(inner(features))


def normalize_features(x):
    mean = ops.mean(x, axis=-1, keepdims=True)
    variance = ops.var(x, axis=-1, keepdims=True)
    return (x - mean) / ops.sqrt(variance + 1e-6)
