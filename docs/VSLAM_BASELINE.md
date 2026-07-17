# Visual SLAM numerical baseline

Deterministic acceptance baseline for the SLAM geometry stack. Every
substantial change must keep this green:

```bash
export KERAS_BACKEND=jax
export JAX_PLATFORMS=cpu
python3 scripts/vslam/run_baseline.py --mode quick
python3 scripts/vslam/run_baseline.py --mode full
```

Both modes run offline, print a compact metric table, write
machine-readable results to `artifacts/vslam_baseline_results.json`,
and exit nonzero when any check fails.

## Files

- `scripts/vslam/fixtures.py` — the five fixture definitions (seeds,
  sizes, noise levels) shared by both scripts.
- `scripts/vslam/create_baseline.py` — recomputes the OpenCV/SciPy
  reference metrics and rewrites the committed baseline. Run only when
  a baseline update is justified (see policy below).
- `scripts/vslam/run_baseline.py` — evaluates the current PAZ
  implementation against the committed baseline.
- `tests/slam/data/vslam_baseline_v1.json` — seeds, fixture
  dimensions, dependency versions, git commit, fixture checksums,
  reference metrics, and tolerances.
- `tests/slam/data/vslam_baseline_v1.npz` — the fixture arrays used to
  detect generator drift through checksums.

Fixtures are generated in float64 (`tests/slam/synthetic.py` enables
jax x64); production code stays dtype-generic and float32 by default.

## Checks

Quick mode:

- Fixture checksums match the committed npz (generator drift guard).
- Noise-free two-view geometry against hard ground-truth thresholds
  (median Sampson < 1e-5 px, rotation < 0.05 deg, translation
  direction < 0.1 deg, triangulation RMSE < 1e-4 px, DLT PnP rotation
  < 0.05 deg and translation < 1e-4).
- Noisy two-view RANSAC and PnP-RANSAC against the OpenCV/SciPy
  reference: precision/recall within 0.03, pose errors within
  1.1x the reference error plus a small absolute slack, refined
  reprojection RMSE within 5% of the SciPy reference.

Full mode adds Schur-complement bundle adjustment on the six-pose
fixture (final robust cost below initial, final RMSE < 0.75 px and
within 1.1x the SciPy reference, pose errors within 1.1x the
reference) and the stereo-sequence trajectory: per-frame jitted
PnP-RANSAC over 40 frames with fixed-capacity visibility masks
(ATE RMSE < 0.03 m, median reprojection < 1 px, no NaN, exactly one
compilation), plus informational RPE, drift, and runtime numbers.

## Reference values (cv2 4.7.0, scipy 1.13.0, numpy 1.26.4)

At baseline creation the PAZ implementation achieved, versus the
reference protocol on identical fixtures:

```text
noisy two-view   rotation error   0.263 deg (reference 0.360)
noisy two-view   direction error  0.345 deg (reference 1.216)
PnP-RANSAC       rotation error   0.052 deg (reference 0.052)
PnP-RANSAC       translation      4.9 mm    (reference 4.9 mm)
refined RMSE     0.658 px         (reference 0.658 px)
bundle RMSE      31.94 -> 0.5738  (reference 31.94 -> 0.5738 px)
stereo ATE RMSE  5.5 mm           (reference per-frame PnP 5.5 mm,
                                   bound 30 mm)
```

The stereo trajectory mirrors the reference protocol: RANSAC followed
by an explicit refinement on the consensus set. Without the
refinement step one frame's pose stops 65 mm short of convergence
from a poor minimal hypothesis, which is why the protocol includes
it (the reference's solvePnPRansac + solvePnP does the same).

## Baseline update policy

A committed baseline may be regenerated only when the previous
reference was demonstrably wrong, or an algorithm intentionally
improved, and only together with documentation of the old and new
metrics in the commit or pull-request description. Never update it to
make a regression disappear.
