# Learning to walk

A JAX / mjx-warp / Keras re-implementation of the unitree_rl_lab robust
velocity task for the Unitree G1 (29 DoF), trained with PPO as in rsl_rl.
Task, rewards, observations, randomization, pushes, curricula and PPO
hyperparameters follow the reference configuration; the deliberate
differences are listed below and the path that led here is in
`LESSONS.md`.

## Training

Two processes, one per GPU, as the reference's two ranks of 4096
environments. Both are needed: the batch size sets the adaptive-KL
learning rate, and a single 4096-environment process stalls the survival
transition.

```bash
python learn_to_walk.py --num_processes 2 --process_id 0 &
python learn_to_walk.py --num_processes 2 --process_id 1 &
```

Do not set `CUDA_VISIBLE_DEVICES`; each process picks its GPU from
`--process_id`. Logs go to `experiments/<timestamp>_walk/training.csv`,
checkpoints to `checkpoints/` next to it every `--save_interval`
iterations. Expected curve (2 x 4096, entropy 0.01): mean episode length
~330 at iteration 1000, full 1000-step episodes and the command
curriculum unlocking from ~4,300, max speed 1.0 m/s and terrain level ~4
by ~10,000. The original reached the same stages at 7,500 and 10,000.

## Evaluation

```bash
MUJOCO_GL=egl python play.py --checkpoint experiments/<run>/checkpoints --forward 0.5
python bench/evaluate_policy.py --keras experiments/<run>/checkpoints --iteration 20000
```

`play.py` renders a video under a fixed command. `bench/evaluate_policy.py`
runs a checkpoint of ours (`--keras`) or a converted reference checkpoint
(`--isaac`, an npz of the rsl_rl state dict) through the full training
environment and reports episode length, per-term rewards in our units and
in the reference's episodic units, termination causes, push-to-death
timing and per-terrain survival. Use it before any physics change: the
reference's own checkpoints must survive here as long as they do in PhysX
(model_1000: 270 vs 303 steps).

## Files

- `learn_to_walk.py` trains; `play.py` renders; `bench/` evaluates.
- `simulation/common.py`, `simulation/robust.py` build the environment:
  reset, step, rewards, pushes, terrain curriculum, divergence guard.
- `rewards.py` holds the reward terms and the reference weights.
- `terrain.py` builds the 9 x 21 tile terrain, one heightfield per tile.
- `randomize.py` draws the per-environment physics randomization.
- `ppo.py`, `rollout.py`, `networks.py`, `distributed.py` implement PPO
  with one process per GPU.
- `robots/` and `assets/` hold the G1 model with the reference actuation.

## Terrain: one heightfield per tile

mjx warp collides heightfields in float32 in the field's own frame. A
single 200 m field cannot resolve the 5 mm foot spheres far from its
origin: joints exploded from benign states at a rate growing from zero at
the centre column to 0.5% of steps at the outer tiles, and the reference's
model_1000 policy survived 134 steps here against 322 in PhysX. With one
field per 8 m tile every contact stays within 4 m of a field origin, the
explosions vanish and the same policy survives as long as in PhysX.

## Divergence guard

A step whose state is non-finite or whose |reward| exceeds 10 (healthy
steps stay within -0.25..0.06) is scored zero and ends its episode
without a bootstrap; the `divergences` log column counts them. A NaN
state would otherwise poison every parameter through the gradient clip,
and a single -700 reward step, seen about once per 2M steps when a joint
is kicked past its limit, inflates the advantage spread and collapses the
adaptive learning rate for hundreds of iterations.

## Differences from the reference

- Foot velocities and the root velocity are read at frame origins;
  IsaacLab reads link centers of mass.
- Undesired contacts are read from the last substep; foot contacts pool
  the last three substeps as the reference does.
- The joint acceleration penalty differences velocities over one control
  step; IsaacLab over one physics substep (about 2.3x smaller numbers
  under this term's small weight).
- PhysX clamps joint velocities at `velocity_limit_sim`; MuJoCo has no
  clamp. Emulating it changed neither survival nor the physics
  statistics of the reference policy here, so it is left out.
- Pyramid slope tiles use a linear ramp; IsaacLab's profile is bilinear.
- Non-foot collision shapes are capsules; the reference carries convex
  mesh hulls with self-collision. Feet are the same four corner spheres.
- The waist roll and pitch pivots sit 1 cm apart along z in the MJCF
  where the training URDF makes them coincident; link geometry and
  inertials match the URDF.
- The explicit Euler integrator is kept where PhysX solves the drives
  implicitly.
- The learned action noise is read through a floor of 0.01; rsl_rl has
  none and a negative deviation is a documented crash there.

## Tests

```bash
JAX_PLATFORMS=cpu pytest examples/rl_gym
```
