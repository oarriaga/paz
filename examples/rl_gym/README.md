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
- The action noise is parameterized as `log_stdv`, so the standard
  deviation stays positive; rsl_rl learns the deviation directly, which is
  a documented crash source.
- Divergence handling above; the reference relied on external restarts.

## Evaluation

```bash
MUJOCO_GL=egl python play.py --checkpoint experiments/<run>/checkpoints
```

## Tests

```bash
JAX_PLATFORMS=cpu pytest examples/rl_gym
```
