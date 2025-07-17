import keras
from keras import ops

backbone_to_model = {
    "convnext_tiny": keras.applications.ConvNeXtTiny,
    "resnet50": keras.applications.ResNet50,
}


def correlate(x, kernel_size=(5, 5)):
    B, H, W, C = ops.shape(x)
    x = ops.relu(x)
    x = identity = ops.normalize(x, axis=-1)
    patches_args = (kernel_size, (1, 1), 1, "same")
    patches = ops.image.extract_patches(x, *patches_args)
    patches = ops.reshape(patches, (B, H, W, *kernel_size, C))
    identity = ops.expand_dims(identity, [3, 4])
    correlations = patches * identity
    return correlations


def block_3D(x, filters, kernel_size, use_bias=False):
    kwargs = {"use_bias": use_bias, "padding": "valid"}
    x = keras.layers.Conv3D(filters, (1, kernel_size, kernel_size), **kwargs)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)


def block_2D(x, filters, kernel_size=1, use_bias=False, activation="relu"):
    kwargs = {"use_bias": use_bias, "padding": "valid"}
    x = keras.layers.Conv2D(filters, kernel_size, **kwargs)(x)
    x = keras.layers.BatchNormalization()(x)
    if activation is not None:
        x = keras.layers.Activation(activation)(x)
    return x


def self_correlate(x, filters=[64, 64, 64, 640], kernel_size=3, bias=False):
    x = correlate(x, kernel_size=(5, 5))
    B, H, W, U, V, C = ops.shape(x)
    x = ops.reshape(x, (B, H * W, U * V, C))
    x = block_2D(x, filters[0], 1, bias, "relu")
    x = ops.reshape(x, (B, H * W, U, V, filters[0]))
    x = self_block_3D(x, filters[1], kernel_size, bias)
    x = self_block_3D(x, filters[2], kernel_size, bias)
    x = ops.reshape(x, (B, H, W, filters[2]))
    x = block_2D(x, filters[3], 1, bias, None)
    return x


def self_block_3D(x, filters, kernel_size, activation="relu"):
    kwargs = {"padding": "same", "use_bias": False}
    x = keras.layers.Conv3D(filters, kernel_size, **kwargs)(x)
    x = keras.layers.BatchNormalization()(x)
    if activation is not None:
        x = keras.layers.Activation(activation)(x)
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


def Refiner(x, kernel_sizes=[(3, 3), (3, 3)], filters=[16, 1]):
    x = inputs = keras.Input(shape=x.shape[1:])
    num_layers = len(kernel_sizes)
    for layer_arg in range(num_layers):
        intro_filters = 1 if layer_arg == 0 else filters[layer_arg - 1]
        outro_filters = filters[layer_arg]
        kernel_size = kernel_sizes[layer_arg]
        x = separable_conv4D(x, intro_filters, outro_filters, *kernel_size)
        if layer_arg < num_layers - 1:
            x = keras.layers.ReLU()(x)
    return keras.Model(inputs, x)


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
    return ops.expand_dims(attention, axis=2)


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


def cross_correlation_attention(
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


def RENet(backbone="resnet_50", weights="imagenet"):
    backbone_kwargs = {"include_top": False, "weights": weights}
    backbone = backbone_to_model[backbone](backbone_kwargs)
    embeder = Embeder(backbone.output_shape[-1])
    refiner = Refiner(backbone.output_shape[-1])

    def encode(x):
        features = backbone(x)
        residual = self_correlate(features)
        features = features + residual
        return ops.relu(features)

    queries = keras.layers.Input("queries")
    support = keras.layers.Input("support")
    encoded_queries = encode(queries)
    encoded_support = encode(support)

    similarity_matrix, queries = cross_correlation_attention(
        encoded_queries,
        encoded_support,
        embeder,
        refiner,
        num_ways,
        num_shots,
        num_queries,
        temperature,
        temperature_attn,
    )
    class_queries = keras.layers.Dense(num_classes)(queries)
    return keras.Model(
        inputs=[queries, support],
        outputs=[similarity_matrix, class_queries],
        name="RENet",
    )
