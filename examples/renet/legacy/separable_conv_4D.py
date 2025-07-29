import keras
from keras import ops


def block(x, filters, kernel_size):
    conv_kwargs = {"padding": "same", "use_bias": False}
    x = keras.layers.Conv3D(filters, kernel_size, **conv_kwargs)(x)
    x = keras.layers.BatchNormalization()(x)
    return x


def project(x, filters):
    x = keras.layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    return x


def separable_conv4D(x, intro_filters, outro_filters, kernel_H, kernel_W):
    _, U, V, H, W, C = x.shape
    x = ops.reshape(x, (-1, U, V, H * W, C))
    x = block(x, intro_filters, (kernel_H, kernel_W, 1))
    x = keras.layers.ReLU(x)
    x = ops.reshape(x, (-1, U * V, H, W, intro_filters))
    x = block(x, intro_filters, (1, kernel_H, kernel_W))
    if intro_filters != outro_filters:
        x = ops.reshape(x, (-1, U * V * H, W, intro_filters))
        x = project(x, outro_filters)
    filters = outro_filters if intro_filters != outro_filters else intro_filters
    x = ops.reshape(x, (-1, U, V, H, W, filters))
    return x


def SharedCCA(x, kernel_sizes=[(3, 3), (3, 3)], filters=[16, 1]):
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


def block_CCA(x, kernel_sizes=[(3, 3), (3, 3)], filters=[16, 1]):
    shared_model = SharedCCA(x, kernel_sizes, filters)
    x1 = shared_model(x)
    axes_permutation = (0, 3, 4, 1, 2, 5)
    x_transposed = ops.transpose(x, axes=axes_permutation)
    x2_processed = shared_model(x_transposed)
    x2 = ops.transpose(x2_processed, axes=axes_permutation)
    return keras.layers.Add()([x1, x2])


def gaussian_normalize(x, axis, epsilon=1e-5):
    """Normalizes input to have zero mean and unit variance across an axis."""
    mean = ops.mean(x, axis=axis, keepdims=True)
    variance = ops.var(x, axis=axis, keepdims=True)
    return (x - mean) / ops.sqrt(variance + epsilon)


def normalize_feature(self, x):
    """Shifts channel activations by the channel mean."""
    return x - ops.mean(x, axis=-1, keepdims=True)


def embed(x, axis, cca_1x1, repeats):
    x = cca_1x1(x)
    x = ops.normalize(x, axis=-1)
    x = ops.expand_dims(x, axis)
    x = ops.repeat(x, repeats, axis)


def compute_correlations_4D(support, queries, cca_1x1):
    """Computes the 4D correlation map between support and query sets."""
    num_queries, num_ways = ops.shape(queries)[0], ops.shape(support)[0]
    support = embed(support, 0, cca_1x1, num_queries)
    queries = embed(queries, 1, cca_1x1, num_ways)
    similarity_map = ops.einsum("qnijc,qnklc->qnijkl", support, queries)
    return similarity_map


def average_embeddings(support, queries, num_queries, num_ways, num_shots):
    support_shape = (num_queries, num_ways, num_shots, *ops.shape(support)[2:])
    support = ops.reshape(support, support_shape)
    support = ops.mean(support, axis=2)
    # The original code reshapes queries, which is unusual if query shot=1.
    queries_shape = (num_queries, num_ways, num_shots, *ops.shape(queries)[2:])
    queries = ops.reshape(queries, queries_shape)
    queries = ops.mean(queries, axis=2)
    return support, queries


def compute_attention(axis, sum_axes, correlations, shape, temperature):
    (num_queries, num_ways, Hs, Ws, Hq, Wq) = shape
    _shape = (num_queries, num_ways, Hs * Ws, Hq, Wq)
    correlations = gaussian_normalize(ops.reshape(correlations, _shape), axis)
    correlations = ops.softmax(correlations / temperature, axis)
    attention = ops.sum(ops.reshape(correlations, shape), axis=sum_axes)
    return ops.expand_dims(attention, axis=2)


def cosine_similarity(support_attended, queries_attended):
    support_normalized = ops.normalize(support_attended, axis=-1)
    queries_normalized = ops.normalize(queries_attended, axis=-1)
    similarities = ops.sum(support_normalized * queries_normalized, -1)
    return similarities


class CrossCorrelationAttention(keras.layers.Layer):
    def __init__(
        self, num_ways, num_shots, temperature, temperature_attn, **kwargs
    ):
        super().__init__(**kwargs)
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.temperature = temperature
        self.temperature_attn = temperature_attn

    def build(self, input_shape):
        spt_shape, _ = input_shape
        feat_dim = spt_shape[-1]

        self.cca_1x1 = keras.Sequential(
            [
                keras.layers.Input(shape=(None, None, feat_dim)),  # H, W, C
                keras.layers.Conv2D(64, kernel_size=1, use_bias=False),
                keras.layers.BatchNormalization(axis=-1),
                keras.layers.ReLU(),
            ],
        )
        super().build(input_shape)

    def call(self, inputs):
        support, queries = inputs
        support = ops.squeeze(support, axis=0)
        support = normalize_feature(support)
        queries = normalize_feature(queries)
        correlations = compute_correlations_4D(support, queries, self.cca_1x1)
        shape = num_queries, num_ways, Hs, Ws, Hq, Wq = ops.shape(correlations)

        correlations = ops.reshape(correlations, (-1, Hs, Ws, Hq, Wq, 1))
        correlations = block_CCA(correlations)  # Refine correlations with CCA
        attention_args = (correlations, shape, self.temperature_attn)
        support_attention = compute_attention(2, [4, 5], *attention_args)
        queries_attention = compute_attention(4, [2, 3], *attention_args)
        support_attended = support_attention * ops.expand_dims(support, axis=0)
        queries_attended = queries_attention * ops.expand_dims(queries, axis=1)

        if self.num_shots > 1:
            shapes = (num_queries, self.num_ways, self.num_shots)
            embeddings = (support_attended, queries_attended)
            averages = average_embeddings(*embeddings, *shapes)
            support_attended, queries_attended = averages

        support_attended = ops.mean(support_attended, axis=[-2, -1])
        queries_attended = ops.mean(queries_attended, axis=[-2, -1])
        queries_pooled = ops.mean(queries, axis=[-2, -1])
        similarities = cosine_similarity(support_attended, queries_attended)
        return similarities / self.temperature, queries_pooled


def self_correlation(x, kernel_size=(5, 5), padding=2):
    B, H, W, C = ops.shape(x)
    x = ops.relu(x)
    x = identity = ops.normalize(x, axis=-1)
    x = keras.layers.ZeroPadding2D(padding)(identity)
    patches_args = ((1, *kernel_size, 1), (1, 1, 1, 1), (1, 1, 1, 1), "valid")
    patches = ops.image.extract_patches(x, *patches_args)
    patches = ops.reshape(patches, (B, H, W, *kernel_size, C))
    identity = ops.expand_dims(ops.expand_dims(identity, axis=-2), axis=-2)
    correlations = patches * identity
    return correlations


def block_SCR(x, filters=[64, 64, 64, 640], kernel_size=3, use_bias=False):
    B, H, W, U, V, C = ops.shape(x)
    x = ops.reshape(x, (B, H * W, U * V, C))
    x = keras.layers.Conv2D(filters[0], 1, use_bias=use_bias, padding="valid")(
        x
    )
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = ops.reshape(x, (B, H * W, U, V, filters[0]))
    x = keras.layers.Conv3D(
        filters[1],
        (1, kernel_size, kernel_size),
        (1, 1, 1),
        use_bias=use_bias,
        padding="valid",
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv3D(
        filters[2],
        (1, kernel_size, kernel_size),
        (1, 1, 1),
        use_bias=use_bias,
        padding="valid",
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = ops.reshape(x, (B, H, W, filters[2]))

    x = keras.layers.Conv2D(
        filters[3],
        1,
        use_bias=use_bias,
        padding="valid",
        name="scr_conv2d_out",
    )(x)
    x = keras.layers.BatchNormalization()(x)
    return x


class SCRLayer(nn.Module):
    def __init__(self, planes=[640, 64, 64, 64, 640]):
        super(SCRLayer, self).__init__()
        kernel_size = (5, 5)
        padding = 2
        stride = (1, 1, 1)
        self.model = nn.Sequential(
            SelfCorrelationComputation(
                kernel_size=kernel_size, padding=padding
            ),
            SCR(planes=planes, stride=stride),
        )

    def forward(self, x):  # b/ewsq c h w
        return self.model(x)


def RENet(feat_dim, lambda_epi, temperature, temperature_attn, num_classes):
    fc = nn.Linear(feat_dim, num_classes)
    scr_layer = SCRLayer(planes=[feat_dim, 64, 64, 64, feat_dim])
    cca_layer = CCALayer(
        feat_dim,
        way_num,
        shot_num,
        query_num,
        temperature,
        temperature_attn,
    )

    def encode(self, x):
        x = emb_func(x)
        identity = x
        x = scr_layer(x)
        x = x + identity
        x = ops.relu(x)
        return x

    def set_forward_loss(batch):
        (
            ep_images,  # ew(qs) c h w
            ep_global_targets,
            g_images,  # b c h w
            g_global_targets,
        ) = batch  # RENet uses both episode and general dataloaders
        # [e x w x (q+s)] -> [e x w x q]
        ep_global_targets_qry = ep_global_targets[..., : self.query_num]

        # extract features
        ep_feat = encode(ep_images)
        g_feat = encode(g_images)  # [128, 640, 5, 5]

        # CCA for ep_images
        support_feat, query_feat, support_target, query_target = (
            self.split_by_episode(ep_feat, mode=2)
        )
        # ws c h w ; wq c h w
        _, _, c, h, w = support_feat.shape
        support_feat = support_feat.reshape([-1, c, h, w])
        query_feat = query_feat.reshape([-1, c, h, w])
        logits, qry_pooled = self.cca_layer(support_feat, query_feat)

        abs_logits = self.fc(qry_pooled)
        epi_loss = self.loss_func(logits, query_target.reshape(-1))
        abs_loss = self.loss_func(abs_logits, ep_global_targets_qry.reshape(-1))

        # FC for g_images
        g_feat = g_feat.mean(dim=[-1, -2])
        logits_aux = self.fc(g_feat)
        aux_loss = self.loss_func(logits_aux, g_global_targets.reshape(-1))
        aux_loss = aux_loss + abs_loss
        loss = self.lambda_epi * epi_loss + aux_loss

        return logits
