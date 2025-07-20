import os

os.environ["KERAS_BACKEND"] = "jax"
import keras
from keras import ops

backbone_to_model = {
    "convnext_tiny": keras.applications.ConvNeXtTiny,
    "resnet50": keras.applications.ResNet50,
}


class Patches(keras.layers.Layer):
    """A stable wrapper layer for the extract_patches operation."""

    def __init__(self, size, strides, dilation_rate, padding, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.padding = padding

    def call(self, x):
        return ops.image.extract_patches(
            x,
            size=self.size,
            strides=self.strides,
            dilation_rate=self.dilation_rate,
            padding=self.padding,
        )


def correlate(x, kernel_size=(5, 5)):
    B, H, W, C = ops.shape(x)
    x = ops.relu(x)
    x = identity = ops.normalize(x, axis=-1)
    patches = Patches(kernel_size, (1, 1), 1, "same")(x)
    print(patches.shape)
    patches = ops.reshape(patches, (B, H, W, *kernel_size, C))
    identity = ops.expand_dims(identity, [3, 4])
    correlations = patches * identity
    return correlations


def correlate_old(x, kernel_size=(5, 5)):
    B, H, W, C = ops.shape(x)
    x = ops.relu(x)
    x = identity = ops.normalize(x, axis=-1)
    # patches_args = (kernel_size, (1, 1), 1, "same")
    patches = ops.image.extract_patches(
        x, size=kernel_size, strides=(1, 1), dilation_rate=1, padding="same"
    )

    patches = ops.reshape(patches, (B, H, W, *kernel_size, C))
    identity = ops.expand_dims(identity, [3, 4])
    correlations = patches * identity
    return correlations


def self_block_2D(x, filters, kernel_size=1, use_bias=False, activation="relu"):
    kwargs = {"use_bias": use_bias, "padding": "valid"}
    x = keras.layers.Conv2D(filters, kernel_size, **kwargs)(x)
    x = keras.layers.BatchNormalization()(x)
    if activation is not None:
        x = keras.layers.Activation(activation)(x)
    return x


def self_block_3D(x, filters, kernel_size, activation="relu"):
    # TODO check if valid does padding correctly
    kwargs = {"padding": "valid", "use_bias": False}
    x = keras.layers.Conv3D(filters, (1, kernel_size, kernel_size), **kwargs)(x)
    x = keras.layers.BatchNormalization()(x)
    if activation is not None:
        x = keras.layers.Activation(activation)(x)
    return x


def self_correlate(x, filters=[64, 64, 64, 640], kernel_size=3, bias=False):
    x = correlate(x, kernel_size=(5, 5))
    print("Correlations", x.shape)
    B, H, W, U, V, C = ops.shape(x)
    x = ops.reshape(x, (B, H * W, U * V, C))
    x = self_block_2D(x, filters[0], 1, bias, "relu")
    x = ops.reshape(x, (B, H * W, U, V, filters[0]))
    x = self_block_3D(x, filters[1], kernel_size)
    x = self_block_3D(x, filters[2], kernel_size)
    print(x.shape, (B, H, W, filters[2]))
    x = ops.reshape(x, (B, H, W, filters[2]))
    x = self_block_2D(x, filters[3], 1, bias, None)
    return x


def block_3D(x, filters, kernel_size, use_bias=False):
    kwargs = {"use_bias": use_bias, "padding": "same"}
    x = keras.layers.Conv3D(filters, kernel_size, **kwargs)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    return x


def project(x, filters):
    x = keras.layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    return x


def separable_conv4D(x, intro_filters, outro_filters, kernel_H, kernel_W):
    _, U, V, H, W, C = x.shape
    x = ops.reshape(x, (-1, U, V, H * W, C))
    x = block_3D(x, intro_filters, (kernel_H, kernel_W, 1), "relu")
    x = ops.reshape(x, (-1, U * V, H, W, intro_filters))
    x = block_3D(x, intro_filters, (1, kernel_H, kernel_W), None)
    if intro_filters != outro_filters:
        x = ops.reshape(x, (-1, U * V * H, W, intro_filters))
        x = project(x, outro_filters)
    filters = outro_filters if intro_filters != outro_filters else intro_filters
    x = ops.reshape(x, (-1, U, V, H, W, filters))
    return x


# def OldRefiner(x, kernel_sizes=[(3, 3), (3, 3)], filters=[16, 1]):
#     x = inputs = keras.Input(shape=x.shape[1:])
#     num_layers = len(kernel_sizes)
#     for layer_arg in range(num_layers):
#         intro_filters = 1 if layer_arg == 0 else filters[layer_arg - 1]
#         outro_filters = filters[layer_arg]
#         kernel_size = kernel_sizes[layer_arg]
#         x = separable_conv4D(x, intro_filters, outro_filters, *kernel_size)
#         if layer_arg < num_layers - 1:
#             x = keras.layers.ReLU()(x)
#     return keras.Model(inputs, x)


def Refiner(input_shape, kernel_sizes=[(3, 3), (3, 3)], filters=[16, 1]):
    """Builds the reusable model for refining correlations."""
    x = inputs = keras.Input(shape=input_shape)
    num_layers = len(kernel_sizes)
    for layer_arg in range(num_layers):
        intro_filters = 1 if layer_arg == 0 else filters[layer_arg - 1]
        outro_filters = filters[layer_arg]
        kernel_size = kernel_sizes[layer_arg]
        x = separable_conv4D(x, intro_filters, outro_filters, *kernel_size)
        if layer_arg < num_layers - 1:
            x = keras.layers.ReLU()(x)
    return keras.Model(inputs, x, name="Refiner")


def refine(x, refiner, kernel_sizes=[(3, 3), (3, 3)], filters=[16, 1]):
    x1 = refiner(x)
    axes_permutation = (0, 3, 4, 1, 2, 5)
    x_transposed = ops.transpose(x, axes=axes_permutation)
    x2_processed = refiner(x_transposed)
    x2 = ops.transpose(x2_processed, axes=axes_permutation)
    return keras.layers.Add()([x1, x2])


def Embeder(feat_dim):
    return keras.Sequential(
        [
            keras.layers.Input(shape=(None, None, feat_dim)),  # H, W, C
            keras.layers.Conv2D(64, kernel_size=1, use_bias=False),
            keras.layers.BatchNormalization(axis=-1),
            keras.layers.ReLU(),
        ],
    )
    return


def normalize_feature(x):
    """Shifts channel activations by the channel mean."""
    return x - ops.mean(x, axis=-1, keepdims=True)


def embed(x, embedder, axis, repeats):
    x = embedder(x)
    x = ops.normalize(x, axis=-1)
    x = ops.expand_dims(x, axis)
    x = ops.repeat(x, repeats, axis)


def compute_similarity(support, queries, embedder):
    """Computes the 4D correlation map between support and query sets."""
    num_queries, num_ways = ops.shape(queries)[0], ops.shape(support)[0]
    support = embed(support, embedder, 0, num_queries)
    queries = embed(queries, embedder, 1, num_ways)
    similarity_map = ops.einsum("qnijc,qnklc->qnijkl", support, queries)
    return similarity_map


def cross_correlate(support, queries, embedder, refiner):
    support = ops.squeeze(support, axis=0)
    support = normalize_feature(support)
    queries = normalize_feature(queries)
    correlations = compute_similarity(support, queries, embedder)
    num_queries, num_ways, Hs, Ws, Hq, Wq = ops.shape(correlations)
    correlations = ops.reshape(correlations, (-1, Hs, Ws, Hq, Wq, 1))
    correlations = refine(correlations, refiner)
    return correlations


def gaussian_normalize(x, axis, epsilon=1e-5):
    """Normalizes input to have zero mean and unit variance across an axis."""
    mean = ops.mean(x, axis=axis, keepdims=True)
    variance = ops.var(x, axis=axis, keepdims=True)
    return (x - mean) / ops.sqrt(variance + epsilon)


def attent(axis, sum_axes, correlations, shape, temperature):
    (num_queries, num_ways, Hs, Ws, Hq, Wq) = shape
    if axis == 2:
        _shape = (num_queries, num_ways, Hs * Ws, Hq, Wq)
    else:
        _shape = (num_queries, num_ways, Hs, Ws, Hq * Wq)
    correlations = gaussian_normalize(ops.reshape(correlations, _shape), axis)
    correlations = ops.softmax(correlations / temperature, axis)
    attention = ops.sum(ops.reshape(correlations, shape), axis=sum_axes)
    # return ops.expand_dims(attention, axis=2)
    return ops.expand_dims(attention, axis=-1)


def cosine_similarity(support_attended, queries_attended):
    support_normalized = ops.normalize(support_attended, axis=-1)
    queries_normalized = ops.normalize(queries_attended, axis=-1)
    similarities = ops.sum(support_normalized * queries_normalized, -1)
    return similarities


def average_embeddings(support, queries, num_queries, num_ways, num_shots):
    support_shape = (num_queries, num_ways, num_shots, *ops.shape(support)[2:])
    support = ops.reshape(support, support_shape)
    support = ops.mean(support, axis=2)
    # The original code reshapes queries, which is unusual if query shot=1.
    queries_shape = (num_queries, num_ways, num_shots, *ops.shape(queries)[2:])
    queries = ops.reshape(queries, queries_shape)
    queries = ops.mean(queries, axis=2)
    return support, queries


def cross_correlation_attention_old(
    support,
    queries,
    embedder,
    refiner,
    num_ways,
    num_shots,
    temperature,
    temperature_attn,
):
    squeezed_support = ops.squeeze(support, axis=0)
    num_queries = ops.shape(queries)[0]
    _, Hs, Ws, _ = ops.shape(squeezed_support)
    _, Hq, Wq, _ = ops.shape(queries)
    shape_6d = (num_queries, num_ways, Hs, Ws, Hq, Wq)

    refined_correlations = cross_correlate(support, queries, embedder, refiner)
    attention_args = (refined_correlations, shape_6d, temperature_attn)
    support_attention = attent(2, [4, 5], *attention_args)
    queries_attention = attent(4, [2, 3], *attention_args)
    support_embedded = embedder(squeezed_support)
    queries_embedded = embedder(queries)
    support_attended = support_attention * ops.expand_dims(support_embedded, 0)
    queries_attended = queries_attention * ops.expand_dims(queries_embedded, 1)

    if num_shots > 1:
        support_attended, queries_attended = average_embeddings(
            support_attended, queries_attended, num_queries, num_ways, num_shots
        )
    support_pooled = ops.mean(support_attended, axis=[-2, -1])
    queries_pooled = ops.mean(queries_attended, axis=[-2, -1])
    queries_orig_pooled = ops.mean(queries, axis=[-2, -1])
    similarity_matrix = cosine_similarity(support_pooled, queries_pooled)
    return similarity_matrix / temperature, queries_orig_pooled


def cross_correlation_attention(
    support_features,
    query_features,
    embedder,
    refiner,
    num_ways,
    num_shots,
    temperature,
    temperature_attn,
):
    """
    Refactored to work with flattened support and query feature maps.

    Args:
        support_features: Tensor of shape (S, Hs, Ws, C), where S = way * shot.
        query_features: Tensor of shape (Q, Hq, Wq, C), where Q = way * query.
    """
    # 1. Embed features using the 1x1 Conv block
    support_embedded = embedder(support_features)
    queries_embedded = embedder(query_features)

    # Get shapes from the embedded features
    S, Hs, Ws, C_embed = ops.shape(support_embedded)
    Q, Hq, Wq, _ = ops.shape(queries_embedded)
    shape_6d = (Q, S, Hq, Wq, Hs, Ws)

    # 2. Compute 6D correlation map with a direct einsum
    # L2 normalize along the channel axis for cosine similarity
    spt_norm = ops.normalize(support_embedded, axis=-1)
    qry_norm = ops.normalize(queries_embedded, axis=-1)

    # Correlate every query feature with every support feature
    correlations = ops.einsum("shwc,qijc->qsijhw", spt_norm, qry_norm)

    # 3. Refine the correlation map
    correlations_reshaped = ops.reshape(
        correlations, (Q * S, Hq, Wq, Hs, Ws, 1)
    )
    refined_correlations = refine(correlations_reshaped, refiner)

    # 4. Compute and apply attention
    # attention_args = (refined_correlations, shape_6d, temperature_attn)
    # Note: The axes are swapped because the einsum output is (Q, S, Hq, Wq, Hs, Ws)
    support_attention = attent(
        axis=4,
        sum_axes=[2, 3],
        correlations=refined_correlations,
        shape=shape_6d,
        temperature=temperature_attn,
    )
    queries_attention = attent(
        axis=2,
        sum_axes=[4, 5],
        correlations=refined_correlations,
        shape=shape_6d,
        temperature=temperature_attn,
    )

    support_attended = support_attention * ops.expand_dims(
        support_embedded, axis=0
    )
    queries_attended = queries_attention * ops.expand_dims(
        queries_embedded, axis=1
    )

    # 5. Average for multi-shot scenarios
    if num_shots > 1:
        # Reshape support set to (Q, num_ways, num_shots, H, W, C) to average shots
        spt_att_reshaped = ops.reshape(
            support_attended, (Q, num_ways, num_shots, Hs, Ws, C_embed)
        )
        support_attended = ops.mean(
            spt_att_reshaped, axis=2
        )  # Average along shot dimension

    # 6. Pool features and compute final similarity
    support_pooled = ops.mean(support_attended, axis=[-2, -1])
    queries_pooled = ops.mean(queries_attended, axis=[-2, -1])
    queries_orig_pooled = ops.mean(query_features, axis=[-2, -1])

    similarity_matrix = cosine_similarity(support_pooled, queries_pooled)

    return similarity_matrix / temperature, queries_orig_pooled


def RENet(
    backbone="resnet50",
    weights="imagenet",
    input_shape=(80, 80, 3),
    num_classes=64,
    num_ways=5,
    num_shots=1,
    num_support=5,
    num_queries=1,
    temperature=10.0,
    temperature_attn=5.0,
):

    backbone_kwargs = {
        "include_top": False,
        "weights": weights,
        "input_shape": input_shape,
    }
    backbone = backbone_to_model[backbone](**backbone_kwargs)
    feat_dim = backbone.output_shape[-1]
    feat_shape = backbone.output_shape[1:]  # (H, W, C)

    embedder = Embeder(feat_dim)

    refiner_input_shape = (*feat_shape[:2], *feat_shape[:2], 1)
    refiner = Refiner(input_shape=refiner_input_shape)

    def encode(x_input):
        features = backbone(x_input)
        print("features", features.shape)
        scr_filters = [64, 64, 64, feat_dim]
        residual = self_correlate(features, filters=scr_filters)
        features = keras.layers.Add()([features, residual])
        return keras.layers.ReLU()(features)

    # Define the model inputs
    support_shape = (num_support, *input_shape)
    queries_shape = (num_queries, *input_shape)
    support = keras.layers.Input(support_shape, num_ways, name="support")
    queries = keras.layers.Input(queries_shape, num_ways, name="queries")

    support = ops.reshape(support, (num_ways * num_support, *input_shape))
    queries = ops.reshape(queries, (num_ways * num_queries, *input_shape))
    print("Support", support.shape, "Queries", queries.shape)

    encoded_support = encode(support)
    encoded_queries = encode(queries)
    # encoded_support shapes (25, 3, 3, 2048)
    # encoded_queries shapes (25, 3, 3, 2048)

    # support_for_cca = ops.reshape(
    #     encoded_support, (1, num_ways * num_shots, *feat_shape)
    # )

    # Apply the Cross-Correlation Attention block
    similarity_matrix, queries_pooled = cross_correlation_attention(
        # support=support_for_cca,
        encoded_support,
        encoded_queries,
        embedder=embedder,
        refiner=refiner,
        num_ways=num_ways,
        num_shots=num_shots,
        temperature=temperature,
        temperature_attn=temperature_attn,
    )
    logits = keras.layers.Dense(num_classes)(queries_pooled)
    return keras.Model(
        inputs=[support, queries],
        outputs=[similarity_matrix, logits],
        name="RENet",
    )


renet_model = RENet()
renet_model.summary()
