# Lessons from replicating the G1 walking policy

This file records what it took to reproduce the unitree_rl_lab / IsaacLab /
rsl_rl robust walking run in this example, what was tried and discarded,
and the process lessons. The first attempt (52 commits, PR #436) reached
the result with a hyperparameter compensation; this branch re-applies only
the changes that turned out to be necessary.

## 1. The root cause

mjx warp collides heightfields in float32 in the heightfield's own frame.
The terrain was one 200 m field (9 x 21 tiles of 8 m plus a 20 m border),
so far from its origin the 5 mm foot spheres had about 10 µm of resolution
and the Newton solver produced explosions from perfectly benign states:
joints at 5–20 rad/s jumped past 100 rad/s within one control step, while
CPU MuJoCo integrated the identical states smoothly.

Evidence chain, in the order it was found:

1. The reference's own `model_1000.pt` run inside our full training
   environment survived 134 steps against 322 in PhysX, with 8x our
   policy's divergence rate, 2.7x its roll/pitch rate and 5.9x its
   vertical base velocity, while tracking, gait and deviation terms
   matched within 5%. The same policy was far more jittery here.
2. Replaying captured pre-explosion states: warp exploded 38 of 48, CPU
   MuJoCo none. Randomization, joint friction, armature, 8 or 20 Newton
   iterations, implicitfast, heightfield thickness and CCD settings
   changed nothing; a plane fixed every case that was not already
   mid-explosion.
3. The divergence rate per terrain column was a symmetric V in |x|: 0 per
   10k steps at the centre column, ~50 at |x| = 80 m, for both policies
   and independent of terrain type. The "flat and boxes are worst"
   pattern was only their position at the outer columns.
4. Same reference policy, four simulators (episode length at reference
   iterations 1000 / 5000 / 7000): PhysX 322 / 510 / 824; ours with one
   heightfield 134 / 122 / 137; ours with a plane 293 / 476 / 784; ours
   with per-tile heightfields 278 / – / 765, zero divergences in 4.1M
   steps.
5. Our own policy inside PhysX scored 296 steps against 262 here before
   the fix, and identical per-step rewards in both simulators after it.

Fix: one heightfield geom per tile plus four border boxes, so every
contact is evaluated within 4 m of a field origin. Upstream mujoco-warp
3.12 carries the same float32 kernel (only overflow reporting changed);
this is worth reporting upstream.

Result: with the reference hyperparameters (entropy 0.01, 2 x 4096
environments) one 60k-iteration run transitioned to full episodes at
~4,300 iterations (the original: ~7,500), unlocked 1.0 m/s by 9,500
(original: 10,000) and reached terrain level ~4.2 (original: 4.7).

## 2. What was required

| change | why | evidence |
|---|---|---|
| one heightfield per tile | float32 precision in the field frame | section 1 |
| warp budgets: `njmax` per environment, `naconmax` per batch; no XLA preallocation | warp allocates outside XLA's pool | out-of-memory and dropped constraints otherwise |
| one process per GPU, 2 x 4096 environments | batch size sets the adaptive-KL learning-rate equilibrium | a single 4096 process pins lr ~1.4e-4 and anneals noise 10x slower |
| alive = 1 − fallen, soft joint limits at 90%, foot contact = net force > 1 N over 3 substeps, all ankle links exempt from undesired contact | reference reward semantics | source audit of IsaacLab / unitree_rl_lab |
| additive planar pushes every 3–8 s, zero joint velocities at reset, unclipped position targets, random respawn level past the top | reference reset/push semantics | `push_by_setting_velocity`, `reset_joints_by_scale`, `terrain_levels_vel` |
| terrain generator: per-tile difficulty, full 6 cm rough noise at all levels, boxes ±, 5/8/2/2/4 columns, pinned columns | reference generator | IsaacLab terrain source |
| torso payload −1..3 kg before the 0.9–1.1 mass scale, inertia rescaled | reference events | `add_base_mass`, `recompute_inertia=True` |
| speed curriculum gated on tracking × survival | reference rates the episodic sum over 20 s | the per-step gate unlocked 4x too early and froze a standing policy |
| one minibatch permutation per update | rsl_rl `mini_batch_generator` | zero cost |
| URDF-exact torso and waist inertials | the MJCF shipped the older revision | per-link audit |
| divergence guard on non-finite states and |reward| > 10 | one NaN poisons all parameters; one −700 step collapses the adaptive lr | value-loss spikes to 85.8 observed |

## 3. What was tried and discarded

Every item below was measured on the single 200 m heightfield, where the
explosions dominated everything else, so the conclusions drawn at the time
were wrong even when the measurements were right.

- **Entropy 0.005** instead of the reference 0.01: the only setting that
  transitioned before the fix; it was compensating unlearnable random
  terminations. With per-tile heightfields 0.01 follows the reference.
- **implicitfast integrator**: judged a "net loss" in a training A/B; the
  A/B was measuring the explosion rate, not the integrator.
- **solref 0.15**: tuned on a spawn transient; buried the standing feet
  22 mm. Reverted for the right reason, but it was never the problem.
- **Joint velocity clamp** (PhysX `velocity_limit_sim`): removes the
  runaway joints but leaves survival unchanged once the terrain is fixed.
- **impratio 10, 64 contacts, elliptic cone, mjlab actuation bundle and
  push regime, capsule sole rails**: none addressed the cause; all reverted.
- **Observation normalizer**: the reference has none; runs used it off.
- **Yaw command range growing to 0.2**: the reference registers only the
  linear-velocity curriculum, so yaw stays ±0.1. The original code was
  right and was "fixed" backwards.
- **Footprint-aware spawn height**: compensated explosions at spawn.
  Without it and with per-tile fields, under 1% of rough-tile episodes at
  the hardest level end in their first second.
- **Decorrelated episode counters, postural stabilizer terms, resume
  from checkpoint**: not in the reference and not needed.

## 4. Mechanisms worth remembering

- Batch size sets the adaptive-KL learning rate: the KL per minibatch
  shrinks with more samples, the schedule raises the rate, and the noise
  anneals faster. Halving the batch is not a neutral change.
- A single extreme reward poisons an update twice: it inflates the
  advantage standardization for every sample, and the resulting KL spike
  drives the adaptive learning rate to its floor for hundreds of
  iterations.
- Warp memory: `njmax` is per environment, `naconmax` per batch;
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` is required next to warp.
- Renderer: MuJoCo's near clipping plane scales with the model extent; a
  200 m terrain pushes it past a 3 m camera and clips half the robot.

## 5. Process lessons

- Fixed-action trajectory benches cannot see rare solver failures: ours
  matched the reference to 1e-3 per step while the policy in the loop
  fell twice as fast. Validate physics with the reference's own trained
  policies in the loop and compare episode length, vertical velocity²,
  roll/pitch rate², and the divergence rate per terrain tile.
- When a failure rate is available, correlate it with geometry before
  touching hyperparameters: the V-shape in |x| settled the cause in one
  plot.
- Replay captured pre-failure states in CPU MuJoCo. It separates an
  engine bug from a modelling problem in minutes.
- Compare per-term reward decompositions in identical units. Their
  episodic sums divided by 20 s and our per-step means were the same
  numbers all along once converted.
- Never tune a hyperparameter to compensate an unexplained loss. The
  compensation hid the bug for two days of A/B runs.
- Build the cross-simulator harness on day one and run it before any
  training A/B; a training run is the most expensive and noisiest
  instrument available.
- Keep the change log honest: several "fixes" moved away from the
  reference (yaw range, normalizer, entropy). A per-change table with the
  reference source and the evidence would have exposed them earlier.

## 6. Remaining gap

At equal tracking (0.80 vs 0.82 per step) the mature return is 16 here
against 23–26 in the original. Term by term inside PhysX our policy is
uniformly a little less refined: higher exploration noise (0.47 vs 0.45),
more joint motion, larger deviations. Ranked hypotheses: single-seed
variance (both curves are one seed), the exploration-noise equilibrium,
the substep joint-acceleration definition, undesired-contact pooling. The
next experiment is two more seeds of this exact configuration.
