import keras
from keras import layers
from keras import ops
from examples.dino_object_detection.models.segmentation_head.depthwise_conv_block import (
    DepthwiseConvBlock,
)
from examples.dino_object_detection.models.segmentation_head.mlp_block import MLPBlock


@keras.saving.register_keras_serializable()
class SegmentationHead(layers.Layer):
    def __init__(
        self, in_dim, num_blocks, bottleneck_ratio=1, downsample_ratio=4, **kwargs
    ):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.num_blocks = num_blocks
        self.bottleneck_ratio = bottleneck_ratio
        self.downsample_ratio = downsample_ratio
        self._export_mode = False

        self.interaction_dim = (
            in_dim // bottleneck_ratio if bottleneck_ratio is not None else in_dim
        )

        self.blocks = [
            DepthwiseConvBlock(in_dim, name=f"block_{i}") for i in range(num_blocks)
        ]

        if bottleneck_ratio is None:
            self.spatial_features_proj = layers.Identity(name="spatial_features_proj")
        else:
            self.spatial_features_proj = layers.Dense(
                self.interaction_dim,
                name="spatial_features_proj",
            )

        self.query_features_block = MLPBlock(in_dim, name="query_features_block")

        if bottleneck_ratio is None:
            self.query_features_proj = layers.Identity(name="query_features_proj")
        else:
            self.query_features_proj = layers.Dense(
                self.interaction_dim, name="query_features_proj"
            )

        self.bias = self.add_weight(
            name="bias", shape=(1,), initializer="zeros", trainable=True
        )

    def export(self):
        self._export_mode = True

    def call(
        self, spatial_features, query_features, image_size=None, skip_blocks=False
    ):
        """
        Args:
            spatial_features: (N, C, H, W)
            query_features: List of (N, Q, C)
            image_size: Tuple (H, W)
        """
        # 1. Resize Spatial Features
        # Move to NHWC for resizing: (N, C, H, W) -> (N, H, W, C)
        x = ops.transpose(spatial_features, (0, 2, 3, 1))

        target_size = (
            image_size[0] // self.downsample_ratio,
            image_size[1] // self.downsample_ratio,
        )
        x = ops.image.resize(x, target_size, interpolation="bilinear")

        # Transpose back to NCHW immediately.
        x_nchw = ops.transpose(x, (0, 3, 1, 2))

        if self._export_mode:
            return self._call_export(x_nchw, query_features, skip_blocks)

        return self._call_standard(x_nchw, query_features, skip_blocks)

    def _call_standard(self, x_nchw, query_features, skip_blocks):
        mask_logits = []

        if not skip_blocks:
            for i, block in enumerate(self.blocks):
                if i >= len(query_features):
                    break

                qf = query_features[i]

                # Update spatial features (Input NCHW -> Output NCHW)
                x_nchw = block(x_nchw)

                # Project spatial (Input NCHW -> Output NCHW)
                if isinstance(self.spatial_features_proj, layers.Identity):
                    sp_proj = x_nchw
                else:
                    # NCHW -> NHWC -> Dense -> NCHW
                    x_nhwc = ops.transpose(x_nchw, (0, 2, 3, 1))
                    sp_proj = self.spatial_features_proj(x_nhwc)
                    sp_proj = ops.transpose(sp_proj, (0, 3, 1, 2))

                # Process Query (Input NQC -> Output NQC)
                qf_proc = self.query_features_block(qf)
                qf_proc = self.query_features_proj(qf_proc)

                # Einsum
                # sp_proj is (B, C, H, W), qf_proc is (B, N, C)
                logit = ops.einsum("bchw,bnc->bnhw", sp_proj, qf_proc) + self.bias
                mask_logits.append(logit)
        else:
            sp_proj = x_nchw

            qf_proc = self.query_features_block(query_features[0])
            qf_proc = self.query_features_proj(qf_proc)

            logit = ops.einsum("bchw,bnc->bnhw", sp_proj, qf_proc) + self.bias
            mask_logits.append(logit)

        return mask_logits

    def _call_export(self, x_nchw, query_features, skip_blocks):
        if not skip_blocks:
            for block in self.blocks:
                x_nchw = block(x_nchw)

        if isinstance(self.spatial_features_proj, layers.Identity):
            sp_proj = x_nchw
        else:
            x_nhwc = ops.transpose(x_nchw, (0, 2, 3, 1))
            sp_proj = self.spatial_features_proj(x_nhwc)
            sp_proj = ops.transpose(sp_proj, (0, 3, 1, 2))

        qf_proc = self.query_features_block(query_features[0])
        qf_proc = self.query_features_proj(qf_proc)

        logit = ops.einsum("bchw,bnc->bnhw", sp_proj, qf_proc) + self.bias
        return [logit]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_dim": self.in_dim,
                "num_blocks": self.num_blocks,
                "bottleneck_ratio": self.bottleneck_ratio,
                "downsample_ratio": self.downsample_ratio,
            }
        )
        return config
