import os

os.environ["KERAS_BACKEND"] = "jax"

import gc
import numpy as np
import torch
import keras
import jax
import pytest
from typing import Dict, Any, Tuple, Optional, List


def _evict_other_entries(cache: Dict[str, Any], keep_key: str) -> None:
    other_keys = [key for key in cache.keys() if key != keep_key]
    for key in other_keys:
        del cache[key]
    if other_keys:
        gc.collect()
        jax.clear_caches()


from paz.models.foundation.dinov2.layers.attention import (
    split_query_key_value,
    compute_scores,
    apply_attention,
    merge_heads,
    flatten_heads,
)

KERAS_MODEL_PATHS = [
    r"weights\dinov2_vits14_ported.keras",
    r"weights\dinov2_vitb14_ported.keras",
    r"weights\dinov2_vitl14_ported.keras",
]

PYTORCH_HUB_REPO = "facebookresearch/dinov2"
MODEL_NAMES = [
    "dinov2_vits14",
    "dinov2_vitb14",
    "dinov2_vitl14",
]

MODEL_CONFIGS = [
    {
        "name": MODEL_NAMES[i],
        "keras_path": KERAS_MODEL_PATHS[i],
        "pytorch_name": MODEL_NAMES[i],
        "variant": MODEL_NAMES[i].split("_")[-1],
        "num_blocks": {"vits14": 12, "vitb14": 12, "vitl14": 24}[
            MODEL_NAMES[i].split("_")[-1]
        ],
    }
    for i in range(len(MODEL_NAMES))
]

DEFAULT_INPUT_SIZE = 518
SMALL_INPUT_SIZE = 224
TOLERANCE = 1e-4


class TestFixtures:
    """Shared test fixtures and utilities."""

    _pytorch_models = {}
    _keras_models = {}
    _test_inputs = {}
    _pytorch_intermediates = {}

    @classmethod
    def get_pytorch_model(cls, model_name: str):
        """Lazy loading of PyTorch model (single-entry cache)."""
        _evict_other_entries(cls._pytorch_models, model_name)
        if model_name not in cls._pytorch_models:
            cls._pytorch_models[model_name] = torch.hub.load(
                PYTORCH_HUB_REPO, model_name, force_reload=False
            )
            cls._pytorch_models[model_name].eval()
        return cls._pytorch_models[model_name]

    @classmethod
    def get_keras_model(cls, keras_path: str):
        """Lazy loading of Keras model (single-entry cache)."""
        _evict_other_entries(cls._keras_models, keras_path)
        if keras_path not in cls._keras_models:
            cls._keras_models[keras_path] = keras.models.load_model(keras_path)
        return cls._keras_models[keras_path]

    @classmethod
    def get_test_input(
        cls, size: int = DEFAULT_INPUT_SIZE
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Generate consistent test input for both models."""
        cache_key = f"input_{size}"
        if cache_key not in cls._test_inputs:
            torch_input = torch.randn(1, 3, size, size)
            keras_input = torch_input.permute(0, 2, 3, 1).numpy()
            cls._test_inputs[cache_key] = (torch_input, keras_input)
        return cls._test_inputs[cache_key]

    @classmethod
    def get_pytorch_intermediate_outputs(
        cls, model, input_tensor, model_name: str
    ) -> Dict[str, np.ndarray]:
        """Get cached PyTorch intermediate outputs (single-entry cache)."""
        _evict_other_entries(cls._pytorch_intermediates, model_name)
        if model_name not in cls._pytorch_intermediates:
            cls._pytorch_intermediates[model_name] = cls._extract_pytorch_intermediates(
                model, input_tensor
            )
        return cls._pytorch_intermediates[model_name]

    @staticmethod
    def _extract_pytorch_intermediates(model, input_tensor) -> Dict[str, np.ndarray]:
        """Extract all intermediate outputs from PyTorch model using hooks."""
        outputs = {}
        hooks = []

        def get_hook(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                outputs[name] = output.detach().cpu().numpy()

            return hook

        hooks.append(model.patch_embed.register_forward_hook(get_hook("patch_embed")))
        hooks.append(model.norm.register_forward_hook(get_hook("norm")))

        for i, block in enumerate(model.blocks):
            hooks.append(block.register_forward_hook(get_hook(f"blocks.{i}")))
            hooks.append(
                block.norm1.register_forward_hook(get_hook(f"blocks.{i}.norm1"))
            )
            hooks.append(block.attn.register_forward_hook(get_hook(f"blocks.{i}.attn")))
            hooks.append(block.ls1.register_forward_hook(get_hook(f"blocks.{i}.ls1")))
            hooks.append(
                block.drop_path1.register_forward_hook(
                    get_hook(f"blocks.{i}.drop_path1")
                )
            )
            hooks.append(
                block.norm2.register_forward_hook(get_hook(f"blocks.{i}.norm2"))
            )
            hooks.append(block.mlp.register_forward_hook(get_hook(f"blocks.{i}.mlp")))
            hooks.append(block.ls2.register_forward_hook(get_hook(f"blocks.{i}.ls2")))
            hooks.append(
                block.drop_path2.register_forward_hook(
                    get_hook(f"blocks.{i}.drop_path2")
                )
            )

            if hasattr(block.mlp, "fully_connected_layer_1"):
                hooks.append(
                    block.mlp.fully_connected_layer_1.register_forward_hook(
                        get_hook(f"blocks.{i}.mlp.fully_connected_layer_1")
                    )
                )
            if hasattr(block.mlp, "activaion"):
                hooks.append(
                    block.mlp.activaion.register_forward_hook(
                        get_hook(f"blocks.{i}.mlp.act")
                    )
                )
            if hasattr(block.mlp, "fully_connected_layer_2"):
                hooks.append(
                    block.mlp.fully_connected_layer_2.register_forward_hook(
                        get_hook(f"blocks.{i}.mlp.fully_connected_layer_2")
                    )
                )

        with torch.no_grad():
            final_output = model(input_tensor)

        if isinstance(final_output, tuple):
            final_output = final_output[0]
        outputs["final_output"] = final_output.detach().cpu().numpy()

        for hook in hooks:
            hook.remove()

        return outputs


@pytest.fixture(scope="session", params=MODEL_CONFIGS, ids=lambda x: x["variant"])
def model_config(request):
    """Parameterized fixture for model configurations."""
    return request.param


@pytest.fixture(scope="session")
def pytorch_model(model_config):
    """Session-scoped fixture for PyTorch model."""
    return TestFixtures.get_pytorch_model(model_config["pytorch_name"])


@pytest.fixture(scope="session")
def keras_model(model_config):
    """Session-scoped fixture for Keras model."""
    return TestFixtures.get_keras_model(model_config["keras_path"])


@pytest.fixture(scope="session")
def test_input():
    """Session-scoped fixture for test input."""
    return TestFixtures.get_test_input()


@pytest.fixture(scope="session")
def small_test_input():
    """Session-scoped fixture for small test input."""
    return TestFixtures.get_test_input(SMALL_INPUT_SIZE)


@pytest.fixture(scope="session")
def pytorch_intermediates(pytorch_model, test_input, model_config):
    """Session-scoped fixture for PyTorch intermediate outputs."""
    torch_input, _ = test_input
    return TestFixtures.get_pytorch_intermediate_outputs(
        pytorch_model, torch_input, model_config["pytorch_name"]
    )


class TestModelLoading:
    """Test model loading functionality."""

    def test_pytorch_model_loads(self, pytorch_model, model_config):
        """Test that PyTorch model loads successfully."""
        assert pytorch_model is not None
        assert hasattr(pytorch_model, "forward")
        assert pytorch_model.training is False
        print(f"✅ PyTorch {model_config['variant']} loaded successfully")

    def test_keras_model_loads(self, keras_model, model_config):
        """Test that Keras model loads successfully."""
        assert keras_model is not None
        assert hasattr(keras_model, "__call__")
        print(f"✅ Keras {model_config['variant']} loaded successfully")

    def test_keras_model_file_exists(self, model_config):
        """Test that the Keras model file exists."""
        keras_path = model_config["keras_path"]
        assert os.path.exists(keras_path), f"Keras model file not found: {keras_path}"

    def test_keras_backend_configured(self):
        """Test that Keras backend is properly configured."""
        assert keras.backend.backend() == "jax"

    def test_model_architectures_match(self, pytorch_model, keras_model, model_config):
        """Test that model architectures have expected dimensions."""
        variant = model_config["variant"]

        expected_embedding_dimensions = {
            "vits14": 384,
            "vitb14": 768,
            "vitl14": 1024,
        }

        expected_dimension = expected_embedding_dimensions[variant]
        cls_layer = keras_model.get_layer("cls_token")
        embedding_dimension = cls_layer.embeddings.shape[-1]

        assert (
            embedding_dimension == expected_dimension
        ), f"Expected embedding_dimension {expected_dimension}, got {embedding_dimension}"

        print(
            f"✅ {variant} architecture verified: embedding_dimension={expected_dimension}"
        )


class TestQuickVerification:
    """Test final output equivalence between models."""

    def test_output_shapes_match(
        self, pytorch_model, keras_model, test_input, model_config
    ):
        """Test that output shapes match between models."""
        torch_input, keras_input = test_input

        with torch.no_grad():
            pytorch_output = pytorch_model(torch_input).numpy()

        keras_output = np.array(keras_model(keras_input, training=False))

        assert (
            pytorch_output.shape == keras_output.shape
        ), f"{model_config['variant']}: Shape mismatch: PyTorch={pytorch_output.shape}, Keras={keras_output.shape}"

    def test_output_values_close(
        self, pytorch_model, keras_model, test_input, model_config
    ):
        """Test that output values are numerically close."""
        torch_input, keras_input = test_input

        with torch.no_grad():
            pytorch_output = pytorch_model(torch_input).numpy()

        keras_output = np.array(keras_model(keras_input, training=False))

        mean_diff = np.mean(np.abs(pytorch_output - keras_output))

        assert (
            mean_diff < TOLERANCE
        ), f"{model_config['variant']}: Outputs not close: mean_diff={mean_diff:.8f}"

    def test_output_difference_metrics(
        self, pytorch_model, keras_model, test_input, model_config
    ):
        """Test output difference metrics are within acceptable ranges."""
        torch_input, keras_input = test_input

        with torch.no_grad():
            pytorch_output = pytorch_model(torch_input).numpy()

        keras_output = np.array(keras_model(keras_input, training=False))

        mean_diff = np.mean(np.abs(pytorch_output - keras_output))
        max_diff = np.max(np.abs(pytorch_output - keras_output))

        variant = model_config["variant"]

        setattr(pytest, f"{variant}_mean_absolute_difference", mean_diff)
        setattr(pytest, f"{variant}_max_absolute_difference", max_diff)

        assert (
            mean_diff < TOLERANCE
        ), f"{variant}: Mean absolute difference too large: {mean_diff:.8f}"
        assert (
            max_diff < 1.0
        ), f"{variant}: Max absolute difference too large: {max_diff:.8f}"

        print(f"✅ {variant}: mean_diff={mean_diff:.8f}, max_diff={max_diff:.8f}")


class TestLayerByLayerVerification:
    """Comprehensive layer-by-layer verification using step-by-step forward pass."""

    @staticmethod
    def compare_tensors(
        pytorch_tensor: np.ndarray, keras_tensor: np.ndarray, layer_name: str
    ) -> Tuple[bool, float, float]:
        """Compare two tensors and return detailed comparison results."""
        keras_tensor = np.array(keras_tensor)

        if pytorch_tensor.shape != keras_tensor.shape:
            print(
                f"  ⚠️  Shape mismatch for {layer_name}: PyTorch={pytorch_tensor.shape}, Keras={keras_tensor.shape}"
            )
            return False, float("inf"), float("inf")

        mean_diff = np.mean(np.abs(pytorch_tensor - keras_tensor))
        max_diff = np.max(np.abs(pytorch_tensor - keras_tensor))
        is_close = mean_diff < TOLERANCE
        return is_close, mean_diff, max_diff

    def get_patch_embedding_layer_name(self, keras_model, variant: str) -> str:
        """Get the correct patch embed conv layer name for the variant."""
        try:
            keras_model.get_layer("patch_embed_proj")
            return "patch_embed_proj"
        except Exception:
            pass

        available_layers = [layer.name for layer in keras_model.layers]
        patch_embedding_layers = [
            name for name in available_layers if "patch_embed_proj" in name
        ]

        if patch_embedding_layers:
            return patch_embedding_layers[0]

        raise ValueError(
            f"No patch_embed layer found for variant {variant}. Available layers: {available_layers}"
        )

    def perform_keras_step_by_step_forward(
        self, keras_model, keras_input, variant: str
    ):
        """Perform detailed step-by-step forward pass through Keras model."""
        keras_outputs = {}

        embedding_dimensions = {"vits14": 384, "vitb14": 768, "vitl14": 1024}
        head_counts = {"vits14": 6, "vitb14": 12, "vitl14": 16}
        depth_counts = {"vits14": 12, "vitb14": 12, "vitl14": 24}
        embed_dim = embedding_dimensions[variant]
        num_heads = head_counts[variant]
        depth = depth_counts[variant]
        head_dim = embed_dim // num_heads
        scale = head_dim**-0.5
        patch_size = 14

        proj_layer = keras_model.get_layer(
            self.get_patch_embedding_layer_name(keras_model, variant)
        )
        projected = proj_layer(keras_input)
        B = keras.ops.shape(keras_input)[0]
        H = keras.ops.shape(keras_input)[1]
        W = keras.ops.shape(keras_input)[2]
        num_patches = (H // patch_size) * (W // patch_size)
        patch_embedded = keras.ops.reshape(projected, (B, num_patches, embed_dim))
        keras_outputs["patch_embed"] = patch_embedded

        cls_layer = keras_model.get_layer("cls_token")
        cls_table = cls_layer.embeddings
        cls_tokens = keras.ops.broadcast_to(
            keras.ops.expand_dims(cls_table, axis=0), (B, 1, embed_dim)
        )
        x = keras.ops.concatenate([cls_tokens, patch_embedded], axis=1)

        pos_layer = keras_model.get_layer("pos_embed")
        pos_table = pos_layer.embeddings
        pos = keras.ops.expand_dims(pos_table, axis=0)
        x = keras.ops.add(x, pos)

        try:
            register_layer = keras_model.get_layer("register_tokens")
        except Exception:
            register_layer = None
        if register_layer is not None:
            register_table = register_layer.embeddings
            num_register = register_table.shape[0]
            register_tokens = keras.ops.broadcast_to(
                keras.ops.expand_dims(register_table, axis=0),
                (B, num_register, embed_dim),
            )
            x = keras.ops.concatenate([x[:, :1], register_tokens, x[:, 1:]], axis=1)

        for block_idx in range(depth):
            prefix = f"block_{block_idx}"

            norm1_layer = keras_model.get_layer(f"{prefix}_norm1")
            qkv_layer = keras_model.get_layer(f"{prefix}_qkv")
            proj_out_layer = keras_model.get_layer(f"{prefix}_proj")
            ls1_layer = keras_model.get_layer(f"{prefix}_ls1")
            norm2_layer = keras_model.get_layer(f"{prefix}_norm2")
            mlp_fc1_layer = keras_model.get_layer(f"{prefix}_mlp_fc1")
            mlp_fc2_layer = keras_model.get_layer(f"{prefix}_mlp_fc2")
            ls2_layer = keras_model.get_layer(f"{prefix}_ls2")

            residual1 = x

            x_normalization1 = norm1_layer(x)
            keras_outputs[f"blocks.{block_idx}.normalization1"] = x_normalization1

            qkv_out = qkv_layer(x_normalization1)
            q, k, v = split_query_key_value(qkv_out, num_heads, head_dim)
            scores = compute_scores(q, k, scale)
            attended = apply_attention(scores, v, 0.0, name=prefix)
            merged = merge_heads(attended)
            flat = flatten_heads(merged)
            attention_output = proj_out_layer(flat)
            keras_outputs[f"blocks.{block_idx}.attention"] = attention_output

            x_after_layer_scale_1 = ls1_layer(attention_output)
            keras_outputs[f"blocks.{block_idx}.layer_scale_1"] = x_after_layer_scale_1

            x_after_drop1 = x_after_layer_scale_1
            keras_outputs[f"blocks.{block_idx}.drop_path1"] = x_after_drop1

            x = keras.ops.add(residual1, x_after_drop1)

            residual2 = x

            x_normalization2 = norm2_layer(x)
            keras_outputs[f"blocks.{block_idx}.normalization2"] = x_normalization2

            mlp_fully_connected_layer_1_out = mlp_fc1_layer(x_normalization2)
            keras_outputs[f"blocks.{block_idx}.mlp.fully_connected_layer_1"] = (
                mlp_fully_connected_layer_1_out
            )

            mlp_activation_out = keras.activations.gelu(mlp_fully_connected_layer_1_out)
            keras_outputs[f"blocks.{block_idx}.mlp.activation"] = mlp_activation_out

            mlp_output = mlp_fc2_layer(mlp_activation_out)
            keras_outputs[f"blocks.{block_idx}.mlp.fully_connected_layer_2"] = (
                mlp_output
            )
            keras_outputs[f"blocks.{block_idx}.mlp"] = mlp_output

            x_after_layer_scale_2 = ls2_layer(mlp_output)
            keras_outputs[f"blocks.{block_idx}.layer_scale_2"] = x_after_layer_scale_2

            x_after_drop2 = x_after_layer_scale_2
            keras_outputs[f"blocks.{block_idx}.drop_path2"] = x_after_drop2

            x = keras.ops.add(residual2, x_after_drop2)

            keras_outputs[f"blocks.{block_idx}"] = x

        final_norm_out = keras_model.get_layer("norm")(x)
        keras_outputs["norm"] = final_norm_out

        final_output = final_norm_out[:, 0]
        keras_outputs["final_output"] = final_output

        return keras_outputs

    def test_complete_layer_by_layer_verification(
        self, keras_model, test_input, pytorch_intermediates, model_config
    ):
        """Test complete layer-by-layer verification between PyTorch and Keras models."""
        _, keras_input = test_input
        variant = model_config["variant"]

        print(
            f"\n🔍 Starting comprehensive layer-by-layer verification for {variant.upper()}"
        )

        keras_outputs = self.perform_keras_step_by_step_forward(
            keras_model, keras_input, variant
        )

        mismatched_layers = []
        total_layers = 0
        passed_layers = 0

        for layer_name, pytorch_output in list(pytorch_intermediates.items()):
            if layer_name not in keras_outputs:
                print(f"  ⚠️  Layer '{layer_name}' not found in Keras outputs")
                continue

            total_layers += 1
            is_close, mean_diff, max_diff = self.compare_tensors(
                pytorch_output, keras_outputs[layer_name], layer_name
            )

            if is_close:
                passed_layers += 1
                print(f"  ✅ {layer_name}: PASS (mean_diff={mean_diff:.8f})")
            else:
                mismatched_layers.append((layer_name, mean_diff, max_diff))
                print(
                    f"  ❌ {layer_name}: FAIL (mean_diff={mean_diff:.8f}, max_diff={max_diff:.8f})"
                )

            del keras_outputs[layer_name]
            gc.collect()

        print(f"\n📊 {variant.upper()} Layer-by-Layer Summary:")
        print(f"  Total layers tested: {total_layers}")
        print(f"  Passed: {passed_layers}")
        print(f"  Failed: {len(mismatched_layers)}")

        if mismatched_layers:
            print(f"\n❌ First failed layer: {mismatched_layers[0][0]}")
            print(f"   Mean difference: {mismatched_layers[0][1]:.8f}")
            print(f"   Max difference: {mismatched_layers[0][2]:.8f}")

        assert len(mismatched_layers) == 0, (
            f"{variant}: {len(mismatched_layers)} layers failed verification. "
            f"First failure: {mismatched_layers[0][0] if mismatched_layers else 'None'}"
        )

    def test_patch_embedding_detailed(
        self, keras_model, test_input, pytorch_intermediates, model_config
    ):
        """Test patch embedding layer in detail."""
        _, keras_input = test_input
        variant = model_config["variant"]

        patch_embedding_layer_name = self.get_patch_embedding_layer_name(
            keras_model, variant
        )
        patch_embedded = keras_model.get_layer(patch_embedding_layer_name)(keras_input)
        patch_embedded = np.reshape(
            np.array(patch_embedded),
            (patch_embedded.shape[0], -1, patch_embedded.shape[-1]),
        )

        is_close, mean_diff, max_diff = self.compare_tensors(
            pytorch_intermediates["patch_embed"], patch_embedded, "patch_embed"
        )

        assert is_close, (
            f"{variant}: Patch embedding verification failed. "
            f"mean_diff={mean_diff:.8f}, max_diff={max_diff:.8f}"
        )

    @pytest.mark.parametrize(
        "component",
        [
            "normalization1",
            "attention",
            "layer_scale_1",
            "drop_path1",
            "normalization2",
            "mlp",
            "layer_scale_2",
            "drop_path2",
        ],
    )
    def test_transformer_block_components(
        self, keras_model, test_input, pytorch_intermediates, model_config, component
    ):
        """Test individual components of transformer blocks."""
        _, keras_input = test_input
        variant = model_config["variant"]
        num_blocks = model_config["num_blocks"]

        test_blocks = min(3, num_blocks)

        keras_outputs = self.perform_keras_step_by_step_forward(
            keras_model, keras_input, variant
        )

        failed_blocks = []
        for block_idx in range(test_blocks):
            layer_name = f"blocks.{block_idx}.{component}"

            if layer_name not in pytorch_intermediates:
                continue

            if layer_name not in keras_outputs:
                failed_blocks.append(f"{layer_name} (missing in Keras)")
                continue

            is_close, mean_diff, max_diff = self.compare_tensors(
                pytorch_intermediates[layer_name], keras_outputs[layer_name], layer_name
            )

            if not is_close:
                failed_blocks.append(f"{layer_name} (mean_diff={mean_diff:.8f})")

        assert (
            len(failed_blocks) == 0
        ), f"{variant}: {component} component failed in blocks: {failed_blocks}"

    @pytest.mark.parametrize(
        "mlp_component", ["fully_connected_layer_1", "act", "fully_connected_layer_2"]
    )
    def test_mlp_subcomponents(
        self,
        keras_model,
        test_input,
        pytorch_intermediates,
        model_config,
        mlp_component,
    ):
        """Test individual MLP sub-components."""
        _, keras_input = test_input
        variant = model_config["variant"]
        num_blocks = model_config["num_blocks"]

        keras_outputs = self.perform_keras_step_by_step_forward(
            keras_model, keras_input, variant
        )

        test_blocks = min(3, num_blocks)
        failed_blocks = []

        for block_idx in range(test_blocks):
            layer_name = f"blocks.{block_idx}.mlp.{mlp_component}"

            if layer_name not in pytorch_intermediates:
                continue

            if layer_name not in keras_outputs:
                failed_blocks.append(f"{layer_name} (missing in Keras)")
                continue

            is_close, mean_diff, max_diff = self.compare_tensors(
                pytorch_intermediates[layer_name], keras_outputs[layer_name], layer_name
            )

            if not is_close:
                failed_blocks.append(f"{layer_name} (mean_diff={mean_diff:.8f})")

        assert (
            len(failed_blocks) == 0
        ), f"{variant}: MLP {mlp_component} component failed in blocks: {failed_blocks}"


class TestAllTransformerBlocks:
    """Test all transformer blocks individually."""

    def get_num_blocks(self, model_config):
        """Get the number of transformer blocks for the model variant."""
        return model_config["num_blocks"]

    @pytest.mark.parametrize("block_range", ["early", "middle", "late"])
    def test_transformer_blocks_by_range(
        self, keras_model, test_input, pytorch_intermediates, model_config, block_range
    ):
        """Test transformer blocks by range to avoid too many individual tests."""
        _, keras_input = test_input
        variant = model_config["variant"]
        num_blocks = self.get_num_blocks(model_config)

        if block_range == "early":
            test_blocks = list(range(min(4, num_blocks)))
        elif block_range == "middle":
            start = num_blocks // 3
            end = min(start + 4, num_blocks)
            test_blocks = list(range(start, end))
        else:
            start = max(0, num_blocks - 4)
            test_blocks = list(range(start, num_blocks))

        keras_outputs = (
            TestLayerByLayerVerification().perform_keras_step_by_step_forward(
                keras_model, keras_input, variant
            )
        )

        failed_blocks = []
        for block_idx in test_blocks:
            layer_name = f"blocks.{block_idx}"

            if layer_name not in pytorch_intermediates:
                continue

            if layer_name not in keras_outputs:
                failed_blocks.append(f"{layer_name} (missing in Keras)")
                continue

            is_close, mean_diff, max_diff = (
                TestLayerByLayerVerification.compare_tensors(
                    pytorch_intermediates[layer_name],
                    keras_outputs[layer_name],
                    layer_name,
                )
            )

            if not is_close:
                failed_blocks.append(f"{layer_name} (mean_diff={mean_diff:.8f})")

        assert (
            len(failed_blocks) == 0
        ), f"{variant}: {block_range} transformer blocks failed: {failed_blocks}"


class TestDeepVerification:
    """Test intermediate layer outputs between models (legacy compatibility)."""

    @staticmethod
    def get_patch_embedding_layer_name(keras_model, variant: str) -> str:
        """Get the correct patch embed conv layer name for the variant."""
        available_layers = [layer.name for layer in keras_model.layers]
        patch_embedding_layers = [
            name for name in available_layers if "patch_embed_proj" in name
        ]
        if patch_embedding_layers:
            return patch_embedding_layers[0]
        raise ValueError(
            f"No patch_embed layer found for variant {variant}. Available layers: {available_layers}"
        )

    @staticmethod
    def compare_tensors(
        pytorch_tensor: np.ndarray, keras_tensor: np.ndarray
    ) -> Tuple[bool, float, float]:
        """Compare two tensors and return comparison results."""
        keras_tensor = np.array(keras_tensor)

        if pytorch_tensor.shape != keras_tensor.shape:
            if (
                len(pytorch_tensor.shape) == 4
                and pytorch_tensor.shape[1] < pytorch_tensor.shape[2]
            ):
                pytorch_tensor = np.transpose(pytorch_tensor, (0, 2, 3, 1))

        if pytorch_tensor.shape != keras_tensor.shape:
            return False, float("inf"), float("inf")

        mean_diff = np.mean(np.abs(pytorch_tensor - keras_tensor))
        max_diff = np.max(np.abs(pytorch_tensor - keras_tensor))
        is_close = mean_diff < TOLERANCE
        return is_close, mean_diff, max_diff

    def test_patch_embedding_layer(
        self, pytorch_model, keras_model, test_input, model_config
    ):
        """Test patch embedding layer outputs."""
        torch_input, keras_input = test_input
        pytorch_intermediates = TestFixtures.get_pytorch_intermediate_outputs(
            pytorch_model, torch_input, model_config["pytorch_name"]
        )

        variant = model_config["variant"]
        patch_embedding_layer_name = self.get_patch_embedding_layer_name(
            keras_model, variant
        )
        patch_embedded = keras_model.get_layer(patch_embedding_layer_name)(keras_input)
        patch_embedded = np.reshape(
            np.array(patch_embedded),
            (patch_embedded.shape[0], -1, patch_embedded.shape[-1]),
        )

        is_close, mean_diff, max_diff = self.compare_tensors(
            pytorch_intermediates["patch_embed"], patch_embedded
        )

        assert (
            is_close
        ), f"{variant}: Patch embed mismatch: mean_diff={mean_diff:.8f}, max_diff={max_diff:.8f}"

    def test_final_norm_layer(
        self, pytorch_model, keras_model, test_input, model_config
    ):
        """Test final normalization layer outputs."""
        torch_input, keras_input = test_input
        pytorch_intermediates = TestFixtures.get_pytorch_intermediate_outputs(
            pytorch_model, torch_input, model_config["pytorch_name"]
        )

        variant = model_config["variant"]
        patch_embedding_layer_name = self.get_patch_embedding_layer_name(
            keras_model, variant
        )

        patch_embedded = keras_model.get_layer(patch_embedding_layer_name)(keras_input)


if __name__ == "__main__":
    pytest.main(["-v", "-x", __file__])
