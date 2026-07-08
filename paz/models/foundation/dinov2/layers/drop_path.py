# DropPath lives in paz.layers so DINOv2 and DINOv3 can share it.
from paz.layers import apply_drop_path, build_noise_shape  # noqa: F401
