import keras
from keras import layers, ops
from examples.dino_object_detection.models.transformer_decoder_head.ms_deform_attn import (
    MSDeformAttn,
)
from examples.dino_object_detection.models.transformer_decoder_head.utils import (
    _get_activation_fn,
)


@keras.saving.register_keras_serializable(package="DeformableDETR")
class TransformerDecoderLayer(keras.Model):

    def __init__(
        self,
        d_model,
        sa_nhead,
        ca_nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        group_detr=1,
        num_feature_levels=4,
        dec_n_points=4,
        skip_self_attn=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.sa_nhead = sa_nhead
        self.ca_nhead = ca_nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout
        self.activation_name = activation
        self.normalize_before = normalize_before
        self.group_detr = group_detr
        self.num_feature_levels = num_feature_levels
        self.dec_n_points = dec_n_points
        self.skip_self_attn = skip_self_attn

        self.self_attn = layers.MultiHeadAttention(
            num_heads=sa_nhead, key_dim=d_model // sa_nhead, dropout=dropout
        )
        self.dropout1 = layers.Dropout(dropout)
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)

        self.cross_attn = MSDeformAttn(
            d_model,
            n_levels=num_feature_levels,
            n_heads=ca_nhead,
            n_points=dec_n_points,
        )

        self.nhead = ca_nhead
        self.linear1 = layers.Dense(units=dim_feedforward)
        self.ffn_dropout = layers.Dropout(dropout)
        self.linear2 = layers.Dense(units=d_model)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.norm3 = layers.LayerNormalization(epsilon=1e-5)
        self.dropout2 = layers.Dropout(dropout)
        self.dropout3 = layers.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.group_detr = group_detr

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def build(self, input_shape):
        super().build(input_shape)

    def call(
        self,
        tgt,
        memory,
        training=None,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
        query_sine_embed=None,
        is_first=False,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        memory_list=None,
        memory_mask_list=None,
        spatial_shapes_list=None,  # NEW ARGUMENT
    ):
        bs, num_queries, _ = tgt.shape
        q = k = tgt + query_pos
        v = tgt

        if training:
            bs, num_queries, d_model = ops.shape(q)
            chunk_size = num_queries // self.group_detr
            q = ops.reshape(q, (bs, self.group_detr, chunk_size, d_model))
            q = ops.transpose(q, (1, 0, 2, 3))
            q = ops.reshape(q, (bs * self.group_detr, chunk_size, d_model))
            k = ops.reshape(k, (bs, self.group_detr, chunk_size, d_model))
            k = ops.transpose(k, (1, 0, 2, 3))
            k = ops.reshape(k, (bs * self.group_detr, chunk_size, d_model))
            v = ops.reshape(v, (bs, self.group_detr, chunk_size, d_model))
            v = ops.transpose(v, (1, 0, 2, 3))
            v = ops.reshape(v, (bs * self.group_detr, chunk_size, d_model))

        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = ops.logical_not(tgt_key_padding_mask)

        tgt2 = self.self_attn(
            query=q,
            value=v,
            key=k,
            attention_mask=tgt_mask if tgt_mask is not None else tgt_key_padding_mask,
            training=training,
        )

        if training:
            shape = ops.shape(tgt2)
            total_batch_dim = shape[0]
            seq_len = shape[1]
            embed_dim = shape[2]
            n_chunks = total_batch_dim // bs
            reshaped_tgt = ops.reshape(tgt2, (n_chunks, bs, seq_len, embed_dim))
            transposed_tgt = ops.transpose(reshaped_tgt, (1, 0, 2, 3))
            new_seq_len = n_chunks * seq_len
            tgt2 = ops.reshape(transposed_tgt, (bs, new_seq_len, embed_dim))

        tgt = tgt + self.dropout1(tgt2, training=training)
        tgt = self.norm1(tgt)

        # Pass memory_list to cross_attn
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            reference_points,
            memory,
            spatial_shapes,
            level_start_index,
            memory_key_padding_mask,
            input_flatten_list=memory_list,
            input_padding_mask_list=memory_mask_list,
            input_spatial_shapes_list=spatial_shapes_list,  # NEW ARGUMENT
        )

        tgt = tgt + self.dropout2(tgt2, training=training)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(
            self.ffn_dropout(self.activation(self.linear1(tgt)), training=training)
        )
        tgt = tgt + self.dropout3(tgt2, training=training)
        tgt = self.norm3(tgt)
        return tgt

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "sa_nhead": self.sa_nhead,
                "ca_nhead": self.ca_nhead,
                "dim_feedforward": self.dim_feedforward,
                "dropout": self.dropout_rate,
                "activation": self.activation_name,
                "normalize_before": self.normalize_before,
                "group_detr": self.group_detr,
                "num_feature_levels": self.num_feature_levels,
                "dec_n_points": self.dec_n_points,
                "skip_self_attn": self.skip_self_attn,
            }
        )
        return config
