import os

# os.environ["KERAS_BACKEND"] = "jax"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import numpy as np
import keras

from paz.models.foundation.dinov3.models.vision_transformer import (
    vit_small,
    vit_base,
    vit_large,
    DinoVisionTransformer,
)
from paz.models.foundation.dinov3.layers.attention import (
    SelfAttention,
    CausalSelfAttention,
)
from paz.models.foundation.dinov3.layers.block import (
    SelfAttentionBlock,
    CausalSelfAttentionBlock,
)
from paz.models.foundation.dinov3.layers.ffn_layers import Mlp, SwiGLUFFN
from paz.models.foundation.dinov3.layers.layer_scale import LayerScale
from paz.models.foundation.dinov3.layers.patch_embed import PatchEmbed
from paz.models.foundation.dinov3.layers.rope_position_encoding import (
    RopePositionEmbedding,
)


MODEL_CONSTRUCTORS = {
    "vits16": vit_small,
    "vitb16": vit_base,
    "vitl16": vit_large,
}

MODEL_WEIGHTS_PATHS = {
    "vits16": "dinov3_vits16_ported.keras",
    "vitb16": "dinov3_vitb16_ported.keras",
    "vitl16": "dinov3_vitl16_ported.keras",
}

MODEL_KWARGS = {
    "img_size": 224,
    "patch_size": 16,
    "ffn_layer": "mlp",
    "untie_cls_and_patch_norms": False,
    "norm_layer": "layernorm",
    "layerscale_init": 1e-6,
    "n_storage_tokens": 4,
    "pos_embed_rope_dtype": "float32",
}


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Extract DINOv3 features from an image"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["vits16", "vitb16", "vitl16"],
        default="vits16",
        help="DINOv3 model variant to use (default: vits16)",
    )
    return parser


def load_dinov3_model(model_name, load_weights=False):
    print(f"Instantiating {model_name} model architecture...")
    constructor = MODEL_CONSTRUCTORS[model_name]
    model = constructor(**MODEL_KWARGS)

    print(f"Building model...")
    dummy_input = np.zeros((1, 224, 224, 3), dtype="float32")
    _ = model(dummy_input, training=False)

    if load_weights:
        weights_dir = os.path.expanduser("~/.keras/paz/models/")
        weights_path = os.path.join(
            weights_dir, MODEL_WEIGHTS_PATHS[model_name]
        )

        if not os.path.exists(weights_path):
            print(
                f"Warning: Weights not found at {weights_path}. "
                "Using random initialization."
            )
        else:
            print(f"Loading pretrained weights from {weights_path}...")
            print(
                "Note: Weight loading from saved models is currently experimental."
            )

    print(f"Model ready!")
    return model


def create_dummy_image(img_size=224):
    image = np.random.randn(1, img_size, img_size, 3).astype("float32")
    image = (image - image.min()) / (image.max() - image.min())
    image = image * 2.0 - 1.0
    return image


def extract_features(model, image):
    print(f"\nExtracting features from image with shape: {image.shape}")
    features = model(image, training=False)
    print(f"Features shape: {features.shape}")
    print(f"Features dtype: {features.dtype}")
    print(f"Features min: {np.min(features):.4f}")
    print(f"Features max: {np.max(features):.4f}")
    print(f"Features mean: {np.mean(features):.4f}")
    print(f"Features std: {np.std(features):.4f}")
    return features


def main():
    parser = build_argument_parser()
    args = parser.parse_args()

    print(f"DINOv3 Feature Extraction Example")
    print(f"=" * 50)
    print(f"Model: {args.model}")
    print(f"=" * 50)

    model = load_dinov3_model(args.model)

    dummy_image = create_dummy_image()

    features = extract_features(model, dummy_image)

    print(f"\n{'=' * 50}")
    print(f"Feature extraction completed successfully!")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
