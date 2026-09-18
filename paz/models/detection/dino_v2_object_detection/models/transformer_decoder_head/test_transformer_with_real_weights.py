import os
import sys

import pytest

_PT_DIR = "examples/rf-detr_original_pytorch_implementation"
_PT_ROOT = os.path.join(os.path.dirname(__file__), *([".."] * 6))
if not os.path.isdir(os.path.join(_PT_ROOT, _PT_DIR)):
    pytest.skip("RF-DETR reference unavailable", allow_module_level=True)
pytest.importorskip("torch")

os.environ.setdefault("KERAS_BACKEND", "jax")

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../../../")
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from rfdetr import (
        RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge, RFDETRBase,
    )
    from rfdetr.config import RFDETRBaseConfig
except ImportError:
    rfdetr_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "../../../../../../examples/rf-detr_original_pytorch_implementation",
    ))
    if rfdetr_path not in sys.path:
        sys.path.insert(0, rfdetr_path)
    from rfdetr import (
        RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge, RFDETRBase,
    )
    from rfdetr.config import RFDETRBaseConfig

from transformer_weights_porting_utils import (
    extract_pt_transformer,
    verify_transformer_parity,
)


MODEL_VARIANTS = {
    "Nano": RFDETRNano,
    "Small": RFDETRSmall,
    "Medium": RFDETRMedium,
    "Large": RFDETRLarge,
}

@pytest.mark.parametrize("variant", list(MODEL_VARIANTS.keys()))
def test_transformer_real_weights(variant):
    print(f"\n{'='*60}")
    print(f"Testing Transformer parity for RFDETR {variant}")
    print(f"{'='*60}")
    
    model_cls = MODEL_VARIANTS[variant]
    
    print(f"Loading pretrained {model_cls.__name__} Transformer...")
    pt_transformer = extract_pt_transformer(model_cls)
    
    verify_transformer_parity(pt_transformer, variant)

@pytest.mark.parametrize("config_overrides", [
    {"dec_n_points": 4},
    {"dec_n_points": 8},
    {"sa_nheads": 4, "ca_nheads": 4},
    {"sa_nheads": 16, "ca_nheads": 16},
    {"dec_layers": 2},
    {"hidden_dim": 128},
])
def test_transformer_synthetic_configs(config_overrides):
    print(f"\n{'='*60}")
    print(
        f"Testing Transformer parity for Synthetic Config: {config_overrides}"
    )
    print(f"{'='*60}")
    
    # Create config from base with overrides
    config_dict = RFDETRBaseConfig().model_dump()
    config_dict["pretrain_weights"] = None
    config_dict.update(config_overrides)
    
    config = RFDETRBaseConfig(**config_dict)
    
    print(f"Creating synthetic Transformer with config: {config_overrides}")
    pt_transformer = extract_pt_transformer(RFDETRBase, config=config)
    
    verify_transformer_parity(pt_transformer, f"Synthetic-{config_overrides}")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
