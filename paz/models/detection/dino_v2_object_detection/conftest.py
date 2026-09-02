# The torch-vs-Keras parity tolerances in this tree are calibrated on CPU:
# Keras/JAX runs on CPU here, so letting torch pick up a CUDA device makes
# every comparison a cross-device one and pushes the backbone diff over the
# 1e-4 fallback threshold. Pinning lives here rather than in individual test
# modules so it is set once, before torch initialises its device list.

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
