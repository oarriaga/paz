# Learning to walk

From-scratch PPO training of a robust velocity-tracking walking policy for
the Unitree G1 (29 DoF) on procedurally generated rough terrain, using JAX,
MuJoCo (mjx warp) and Keras. The task, rewards, observations, and PPO
hyperparameters mirror the unitree_rl_lab / rsl_rl reference configuration.

## Files

- `learn_to_walk.py` trains the robust walking policy.
- `play.py` loads a saved actor and writes a rollout video.
- `terrain.py` builds the tiled heightfield with a difficulty curriculum.
- `world.py` compiles robot plus terrain into mjx dynamics.
- `simulation/` environment reset and step, observations, and terminations.
- `rewards.py` reward terms and their reference weights.
- `randomize.py` per-environment physics randomization.
- `rollout.py` experience collection with GAE.
- `ppo.py` the clipped surrogate update with adaptive learning rate.
- `networks.py` Keras actor and critic plus parameter packing.
- `curriculum.py` command speed curriculum.
- `distributed.py` multi-process data parallelism.
- `checkpoint.py` Keras model plus optimizer state checkpoints.

## Training

```bash
python learn_to_walk.py
```

Checkpoints are written every `--save_interval` iterations. Resume with:

```bash
python learn_to_walk.py --load experiments/<run>/checkpoints
```

The saved `max_speed` restores the command curriculum; terrain levels live
in the environment state and re-climb after a resume.

## Multiple GPUs

One process per GPU, as in the reference torchrun setup. Each process
simulates its own `--num_envs` environments; the update runs as one global
program, so minibatch statistics, the KL, and gradients are averaged over
all processes. Launch one process per GPU:

```bash
python learn_to_walk.py --num_processes 2 --process_id 0 &
python learn_to_walk.py --num_processes 2 --process_id 1
```

## Simulator divergence

A non-finite or unphysically fast state after a physics step is treated as
a solver failure rather than an outcome: the step is masked out of the
loss, the episode terminates without a value bootstrap, and the state is
sanitized so nothing poisons the observation history. The `NaN` column in
the training log counts these events.

## Deliberate differences from the reference implementation

- Experience is reshuffled every epoch; rsl_rl draws one permutation per
  update and reuses it across epochs.
- The action noise is learned directly as in rsl_rl, but read through a
  small positive floor; without it a negative deviation is a documented
  crash source. A log parameterization was tried and rejected: it shrank
  the noise fast enough to stall exploration in a standing policy.
- Divergence handling above; the reference relied on external restarts.

## Known remaining differences from the reference

Documented after a four-way source audit (rsl_rl update math, IsaacLab
manager semantics, reward formulas, physics configuration); all judged
minor and left as is:

- Root and foot velocities are taken at the frame origins; IsaacLab uses
  the link centers of mass.
- Foot contact is the last-substep net force; IsaacLab thresholds the
  maximum over a three-substep history.
- The critic sees a push one step later than IsaacLab does.
- Pyramid slope tiles use a linear ramp and a slightly taller platform
  than IsaacLab's bilinear profile.
- Each reset samples a fresh terrain column; IsaacLab pins each
  environment to one column for the whole run.
- The MJCF torso is about 1.8 kg heavier than the training URDF.
- PhysX clamps joint velocities at the actuator limits inside the
  solver; MuJoCo has no such clamp, so unphysical states are instead
  discarded by the divergence guards.
- The explicit Euler integrator stays despite the reference's implicit
  joint drives: an A/B run showed implicitfast slows early balance
  learning by an order of magnitude here.
- The reference trains with full-sole foot collision hulls; this model
  ships four small spheres per foot, a harder support polygon.

## Evaluation

```bash
MUJOCO_GL=egl python play.py --checkpoint experiments/<run>/checkpoints
```

## Tests

```bash
JAX_PLATFORMS=cpu pytest examples/rl_gym
```
