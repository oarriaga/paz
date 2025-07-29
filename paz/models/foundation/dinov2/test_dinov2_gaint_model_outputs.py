import os
import pytest
import numpy as np

TOLERANCE = 1e-4
OUTPUT_DIR = "test_outputs"
KERAS_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "keras_outputs.npz")
PYTORCH_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "pytorch_outputs.npz")


@pytest.fixture(scope="session")
def keras_outputs():
    """Loads pre-computed Keras outputs from disk."""
    if not os.path.exists(KERAS_OUTPUT_PATH):
        pytest.fail(
            f"Keras output file not found: {KERAS_OUTPUT_PATH}. Run generate_outputs.py first."
        )
    return np.load(KERAS_OUTPUT_PATH)


@pytest.fixture(scope="session")
def pytorch_outputs():
    """Loads pre-computed PyTorch outputs from disk."""
    if not os.path.exists(PYTORCH_OUTPUT_PATH):
        pytest.fail(
            f"PyTorch output file not found: {PYTORCH_OUTPUT_PATH}. Run generate_outputs.py first."
        )
    return np.load(PYTORCH_OUTPUT_PATH)


class TestFinalOutputVerification:
    """Tests to compare the final output of both models from saved files."""

    def test_output_values_close(self, pytorch_outputs, keras_outputs):
        pytorch_output = pytorch_outputs["final_output"]
        keras_output = keras_outputs["final_output_from_full_model"]

        mean_diff = np.mean(np.abs(pytorch_output - keras_output))
        print(f"\nFinal output mean difference: {mean_diff:.8f}")

        assert (
            mean_diff < TOLERANCE
        ), f"Final outputs are not close. Mean difference: {mean_diff:.8f}"


class TestLayerByLayerVerification:
    """Comprehensive layer-by-layer comparison from saved files."""

    @staticmethod
    def compare_tensors(
        pytorch_tensor: np.ndarray, keras_tensor: np.ndarray, layer_name: str
    ):
        """Compares two tensors and returns detailed results."""
        mean_diff = np.mean(np.abs(pytorch_tensor - keras_tensor))
        is_close = mean_diff < TOLERANCE
        return is_close, mean_diff

    def test_all_layers_verification(self, keras_outputs, pytorch_outputs):
        """Runs a comprehensive layer-by-layer verification."""
        print("\n🔍 Starting comprehensive verification from saved files for ViT-Giant")

        mismatched_layers = []
        for layer_name in pytorch_outputs.files:
            if layer_name not in keras_outputs.files:
                print(f" ⚠️ {layer_name}: Not found in Keras outputs, skipping.")
                continue

            pytorch_tensor = pytorch_outputs[layer_name]
            keras_tensor = keras_outputs[layer_name]

            is_close, mean_diff = self.compare_tensors(
                pytorch_tensor, keras_tensor, layer_name
            )

            if is_close:
                print(f"   ✅ {layer_name}: PASS (mean_diff={mean_diff:.8f})")
            else:
                mismatched_layers.append((layer_name, mean_diff))
                print(f"   ❌ {layer_name}: FAIL (mean_diff={mean_diff:.8f})")

        assert (
            not mismatched_layers
        ), f"Found {len(mismatched_layers)} mismatched layers."

    @pytest.mark.parametrize("block_idx", [0, 19, 39])
    def test_transformer_block_outputs(self, keras_outputs, pytorch_outputs, block_idx):
        """Verify the output of specific transformer blocks."""
        layer_name = f"blocks.{block_idx}"
        pytorch_tensor = pytorch_outputs[layer_name]
        keras_tensor = keras_outputs[layer_name]

        is_close, mean_diff = self.compare_tensors(
            pytorch_tensor, keras_tensor, layer_name
        )
        assert is_close, f"{layer_name} output mismatch: mean_diff={mean_diff:.8f}"


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
