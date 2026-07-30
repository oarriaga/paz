# Visual SLAM provenance and licensing

The PAZ visual SLAM implementation is a clean-room JAX implementation.
No source code was copied from restrictively licensed SLAM systems.

## Papers and textbooks used

- Hartley, Zisserman: "Multiple View Geometry in Computer Vision",
  2nd edition. Normalized eight-point algorithm (§11.2), essential
  matrix decomposition and cheirality (§9.6), DLT triangulation
  (§12.2), DLT camera resection (§7.1).
- Hartley: "In Defense of the Eight-Point Algorithm", TPAMI 1997
  (point normalization).
- Fischler, Bolles: "Random Sample Consensus", CACM 1981 (RANSAC).
- Barfoot: "State Estimation for Robotics", 2nd edition (SE(3)
  left Jacobian and its inverse, retraction conventions).
- Sola, Deray, Atchuthan: "A micro Lie theory for state estimation in
  robotics", arXiv:1812.01537 (Exp/Log/retract/local_coordinates
  naming and first-order Jacobian identities).
- Lepetit, Moreno-Noguer, Fua: "EPnP: An Accurate O(n) Solution to the
  PnP Problem", IJCV 2009 (referenced for the planned EPnP solver).
- Triggs, McLauchlan, Hartley, Fitzgibbon: "Bundle Adjustment — A
  Modern Synthesis", 2000 (Schur complement structure, planned).
- Umeyama: "Least-squares estimation of transformation parameters
  between two point patterns", TPAMI 1991 (trajectory alignment for
  ATE in the test metrics).
- Sturm et al.: "A Benchmark for the Evaluation of RGB-D SLAM
  Systems", IROS 2012 (ATE/RPE metric definitions).

## External code used as numerical test oracles only

- OpenCV 4.7.0 (Apache-2.0): `findFundamentalMat`, `recoverPose`,
  `triangulatePoints`, `solvePnP`, `solvePnPRansac`. Called only from
  `tests/slam/reference.py` and co-located `*_test.py` files.
- SciPy 1.13.0 (BSD-3-Clause): `scipy.optimize.least_squares` for
  pose-refinement and small bundle-adjustment references. Test-only.

Production modules under `paz/backend` and `paz/slam` do not import
OpenCV or SciPy estimators. Exact oracle versions are recorded by
`tests/slam/reference.py:get_reference_versions` and in the committed
baseline JSON.

## External repositories inspected

None. No SLAM system source code (ORB-SLAM, GPL; OpenVSLAM/stella,
BSD-with-history-caveats; DSO, GPL; BAD-SLAM, etc.) was read or
copied for this implementation. Behavior is derived from the papers
above and validated against the OpenCV/SciPy oracles.

## Repository license

The repository's MIT license (deleted inadvertently by commit
`383ef992`, "Start paz keras 3 port") is restored at `LICENSE` and
declared in `pyproject.toml`. All new SLAM code is contributed under
that MIT license. No GPL, AGPL, non-commercial, or research-only
dependencies were added.
