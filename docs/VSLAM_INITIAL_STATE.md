# Visual SLAM initial repository state

Recorded before any SLAM implementation work, on branch
`feat/visual-slam` created from `paz-jax` at commit `2e467d3e`.

## Environment

```text
python  3.10 (system python3)
jax     0.4.35 (CPU, JAX_PLATFORMS=cpu)
numpy   1.26.4
opencv  4.7.0
scipy   1.13.0
KERAS_BACKEND=jax
```

## Pre-existing test results

```text
pytest -q paz/backend/lie
  1 failed, 86 passed
  FAILED paz/backend/lie/SE3_test.py::test_xyz_rpy_to_SE3
    AttributeError: module paz.backend.lie.SE3 has no attribute
    xyz_rpy_to_SE3 (SE3.py defines xyz_rpy_to_matrix with a TODO to
    rename; pre-existing on paz-jax, unrelated to SLAM work)

pytest -q examples/structure_from_motion/geometry_test.py
  passed (part of the same run: 86 passed overall with lie tests)

pytest -q paz/models/feature
  8 passed, 2 skipped (weight-gated parity tests skip without
  XFEAT_WEIGHTS / LIGHTERGLUE_WEIGHTS environment variables)
```

`pytest -q paz/models/foundation/depth_anything3` was not run here; it
depends on downloaded foundation weights and is unrelated to the SLAM
geometry baseline.

## Conventions confirmed by inspection

- `paz.SE3` operates on 4x4 affine matrices. The se(3) tangent vector
  ordering is `[angular_x, angular_y, angular_z, linear_x, linear_y,
  linear_z]`, pinned by `SE3_test.py::test_get_angular_velocity`.
- `paz.pinhole` projection expects world-to-camera transforms:
  `make_camera_matrix(intrinsics, pose)` computes `[K|0] @ pose`.
- The structure-from-motion example (`examples/structure_from_motion/
  geometry.py`) uses the epipolar convention `x_B^T F x_A = 0` with
  `points1` from image A, and `E = K^T F K`. Its `recover_pose`
  cheirality check only tested depth in the first camera.
- `paz.models.XFeat` extracts at fixed `top_k` inside its jitted core
  and filters dynamically only in the public closure. `LighterGlue`
  pads to fixed `capacity` with masks and `-1` sentinels inside its
  jitted `match`; the public closure slices back to dynamic length.
  Fixed-capacity primitives therefore already exist for a SLAM
  frontend.
- `paz.optimization.minimize` is a first-order optax-based scalar-loss
  driver built on `jax.lax.while_loop`; no second-order least-squares
  solver exists yet.
- No `LICENSE` file and no license metadata existed on `paz-jax`; the
  MIT license file was deleted by commit `383ef992` ("Start paz keras 3
  port") and is restored in this branch.
