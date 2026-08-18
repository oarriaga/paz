# A learned G1 walking policy in MuJoCo

This example replays a `unitree_rl_lab` velocity-tracking policy for the
Unitree G1 29-DoF, trained in IsaacLab under
`Unitree-G1-29dof-Velocity-Robust`, inside MuJoCo. It is a sim-to-sim test:
the policy never saw MuJoCo, so what it does here is a fair proxy for what
it would do on hardware.

The actor runs as pure Keras 3 on the JAX backend, compiled with
`jax.jit`. There is no ONNX runtime anywhere in the loop; the weights are
read straight out of the rsl_rl `model_*.pt` and copied into a functional
Keras model.

The point of the example is measurement. It drives the robot over the same
five terrains it trained on, shoves it with a configurable force, and
reports how often it goes down.

## Install

From the PAZ repository:

    python3 -m pip install -e '.[learning_to_walk]'

`torch` is only used to read the checkpoint. Nothing in the control loop
touches it.

A PlayStation controller is optional. Connect one over USB or Bluetooth to
steer; SDL reads it through `/dev/input/js0`, so no driver setup is needed
beyond pairing it. With no pad connected the robot holds `--speed` forward
for the whole run, which is what makes a terrain measurable.

The demo also needs the G1 MuJoCo plant from a local `unitree_mujoco`
checkout, at `unitree_robots/g1/scene_29dof.xml`. It finds that
automatically when the repository is a sibling of PAZ. For another layout,
pass the location explicitly:

    python3 examples/learning_to_walk/mujoco_demo.py \
        --scene-dir /path/to/unitree_mujoco/unitree_robots/g1

`WALK_SCENE_DIR` can be set instead of passing the flag every time.

## Checkpoints

Training runs go in `unitree_g1_29dof_velocity_robust/` next to this
README, one directory per run, exactly as `unitree_rl_lab` writes them:

    unitree_g1_29dof_velocity_robust/
        2026-08-18_08-19-08/
            model_28100.pt
            params/deploy.yaml
            params/env.yaml

Both scripts pick the newest run's highest iteration by default. Pass
`--checkpoint` to replay a specific one. The directory is gitignored, since
a single run is a couple of hundred megabytes.

## Run

    python3 examples/learning_to_walk/mujoco_demo.py

The actor is compiled with `jax.jit` up front, so the first launch takes a
few seconds. Compiling matters less here than for a larger controller, but
it is still worth an order of magnitude:

    eager   0.72 ms/call CPU   1.94 ms/call GPU
    jitted  0.03 ms/call CPU   0.12 ms/call GPU

A single 29-joint actor at batch one is bound by launch latency rather than
throughput, so the CPU wins per call and leaves the GPU free. The demo
therefore runs on the CPU by default. To override:

    JAX_PLATFORMS=cuda python3 examples/learning_to_walk/mujoco_demo.py

The policy pins `jax_default_matmul_precision` to `float32`. Left at the
XLA default a GPU runs these matmuls as TF32, which moves an action by
about 6e-3 against the checkpoint, roughly a thousand times the error the
CPU makes.

A fixed-length headless run is a useful smoke check:

    python3 examples/learning_to_walk/mujoco_demo.py --headless \
        --steps 4000 --speed 0.5

`--headless` drops the viewer and ignores the pad, so a scripted run always
holds `--speed` and repeats. With the viewer, a connected pad steers and
`--speed` takes over when there is none.

## Controls

| Input | Action |
| --- | --- |
| Left stick up / down | Forward velocity, 1.0 m/s ahead and 0.5 m/s back |
| Left stick left / right | Lateral velocity, up to 0.3 m/s |
| Right stick left / right | Yaw rate, up to 0.2 rad/s |
| Cross | Shove the torso |

Forward tracks well and lateral tracks weakly, but this policy does not
turn: commanded at its trained maximum of 0.2 rad/s it yaws at 0.014 rad/s,
and even at 1.0 rad/s it only reaches 0.107. `track_ang_vel_z` uses
`exp(-error^2 / 0.5^2)` against a command range of only 0.2, so never
turning already scores 0.85 of the available reward and there is almost
nothing to learn from. Widening `ang_vel_z` and narrowing that std are
training changes, not demo ones.

Those limits are the command `limit_ranges` the policy was trained against,
so a fully deflected stick asks for exactly the fastest thing it has seen.
Yaw is the narrow one: this policy only ever trained on 0.2 rad/s, so it
turns slowly no matter how hard the stick is pushed.

The robot stands back up wherever it falls, so a run keeps going and the
final line reports how many times it went down.

## Terrain

`--terrain` picks the ground and `--difficulty` scales it, where 1 is
exactly what the training config specified. The four generated terrains
rebuild the sub-terrains of
`ROBUST_TERRAINS_CFG` as MuJoCo heightfields, keeping the cell size, height
step and grade the training config asked for:

| Name | Training sub-terrain | At difficulty 1 |
| --- | --- | --- |
| `flat` | `MeshPlaneTerrain` | the released ground plane |
| `rough` | `HfRandomUniformTerrain` | rocks of 0 to 6 cm in 1 cm steps, on a 10 cm grid |
| `slope` | `HfPyramidSlopedTerrain` | a 20 % grade falling away from a 2 m platform |
| `inverted_slope` | `HfInvertedPyramidSlopedTerrain` | the same grade climbing away from it |
| `boxes` | `MeshRandomGridTerrain` | a grid of 45 cm boxes, 0 to 5 cm tall |
| `hills` | none, see below | rolling ground 6 cm deep, 2 m down to 25 cm |

Difficulty scales the tallest feature or the grade and nothing else, so the
`rough` height step stays at 1 cm and the box grid stays 45 cm wide at every
setting. Below difficulty 0.084 the rough terrain would round to zero height
levels, so it clamps to a single level rather than dividing by it.

IsaacLab lays its sub-terrains out as 8 by 8 metre tiles. These patches
span 20 metres instead, which keeps the local grade and cell size, what the
feet actually feel, rather than the tile size. The released ground plane
stays underneath the heightfield, so a fast run that reaches the edge of the
patch walks off onto flat ground instead of out of the world.

    python3 examples/learning_to_walk/mujoco_demo.py --terrain boxes
    python3 examples/learning_to_walk/mujoco_demo.py --terrain rough \
        --difficulty 0.5 --seed 3

### Correlated ground

Every terrain in the training mix is spatially uncorrelated. `rough` draws
each 10 cm cell independently of its neighbours and `boxes` draws each box
independently, so neither carries structure wider than one cell or one box.
Real ground does. `hills` sums four octaves of smooth noise, halving the
wavelength and the amplitude each time, which is the standard way to build a
surface with the roughly fractal spectrum natural terrain has.

    python3 examples/learning_to_walk/mujoco_demo.py --terrain hills

Its default relief is 6 cm, the same as `rough` at difficulty 1, so the
comparison isolates shape rather than height. At matched relief the two are
not remotely the same surface:

| Terrain | Relief | Mean local slope | Largest step between cells | Correlation length |
| --- | --- | --- | --- | --- |
| `rough` | 6 cm | 22.8 % | 6.0 cm | 10 cm, one cell |
| `boxes` | 5 cm | 3.8 % | 4.9 cm | 30 cm |
| `hills` | 6 cm | 1.1 % | 0.6 cm | 150 cm |

Which says something about the training terrain as much as about this one:
6 cm steps across 10 cm cells is rubble, not ground, and a policy trained on
it has only ever seen relief arriving as high-frequency noise. `hills` at
equal relief is twenty times gentler underfoot, so it gets hard through
amplitude rather than through steps, and `--difficulty 10` is 60 cm of
relief over 2 m, which is ordinary rough countryside.

`hills` is reported as out of distribution at every difficulty, including
below 1, because no amount of scaling turns an uncorrelated surface into a
correlated one.

### Out of distribution

`--difficulty` is not capped at 1. Above it the terrain keeps scaling
linearly past anything the policy trained on, which is the point: `rough`
holds its 1 cm height step and just stacks more of them, and the slopes keep
the same shape at a steeper grade.

    python3 examples/learning_to_walk/mujoco_demo.py --terrain rough \
        --difficulty 2.5

Both scripts say what a difficulty actually builds, and flag it when it
leaves the training range, so a number in the flag is never the only record
of what was tested:

    Control 50 Hz over physics at 200 Hz, decimation 4
    Terrain rough at difficulty 2.50: rocks up to 15.0 cm in 1.0 cm steps
      OUT OF DISTRIBUTION: 2.50x the 6.0 cm rocks it trained on
    Pushes shove the torso at 400 N for 0.2 s, worth 2.28 m/s
      OUT OF DISTRIBUTION: 2.28x the 1.0 m/s it trained on

A push is measured against the 1.0 m/s velocity kick `push_robot` applied
during training, so both axes of the test report the same way.

## Perturbations

`--push-force` shoves the torso through `xfrc_applied` for 0.2 s, in a
random horizontal direction, on the 3 to 8 second interval the training
config pushed at. On this plant a newton is worth 0.0057 m/s of base
speed, so the 1.0 m/s velocity kick `push_robot` applied during training
lands at about 175 N.

    python3 examples/learning_to_walk/mujoco_demo.py --push-force 300

The pad's cross button shoves on demand whenever the viewer is running.
Holding it keeps shoving, one push per 0.2 s. `--headless` ignores the pad,
so a scripted run only ever takes the scheduled shoves.

An active shove draws an orange arrow into the torso, pointing the way the
force is applied and growing with its magnitude, so a scheduled push is
visible rather than an unexplained stumble. The status line carries the same
number, which is what the headless runs have to go on:

    command [0.5 0. 0.]  height 0.79  tilt 2.1 deg  push  300 N

## Measuring

`evaluate.py` runs the sweep: every terrain, every seed, one command held
throughout, no viewer and no pad. A run ends when the base tilts past
0.8 rad or drops below 0.2 m, the `bad_orientation` and `base_height`
terminations the policy trained under. Left running past that point, the
joint gains keep driving a collapsed robot until MuJoCo blows up.

Each seed draws a starting heading and a set of joint speeds, the way the
training reset did. Without that the deterministic terrains would replay one
identical rollout for every seed.

    python3 examples/learning_to_walk/evaluate.py --seeds 10 --steps 4000

All numbers below hold 0.5 m/s forward for 20 seconds over 10 seeds, at
difficulty 1, against `2026-08-18_08-19-08/model_28100.pt` at 28100
iterations. That run reached the full command curriculum and terrain level
4.8 of 8.

### Terrain alone

Falls out of 10, no pushes, holding 0.5 m/s forward for 20 seconds at
difficulty 1:

| Terrain | Falls | Median distance |
| --- | --- | --- |
| flat | 0/10 | 9.62 m |
| rough | 2/10 | 8.75 m |
| slope | 1/10 | 7.62 m |
| inverted_slope | 0/10 | 9.92 m |
| boxes | 0/10 | 9.56 m |
| hills | 0/10 | 9.61 m |

Nothing the policy trained on troubles it much. A run that stays up covers
the ground it covers on flat, so the rocks, the grade and the boxes cost
almost no speed inside the training range.

Sweeping the height confirms there is no interesting structure below
difficulty 1 either:

| Tallest step | `rough` falls | `boxes` falls |
| --- | --- | --- |
| 1 cm | 0/10 | 0/10 |
| 2 cm | 0/10 | 0/10 |
| 3 cm | 0/10 | 1/10 |
| 4 cm | 1/10 | 0/10 |
| 5 cm | 1/10 | 0/10 |
| 6 cm | 2/10 | -- |

### Pushes

Falls out of 10, holding 0.5 m/s forward over 20 seconds at difficulty 1:

| Push | Base speed | flat | rough | boxes |
| --- | --- | --- | --- | --- |
| 0 N | -- | 0/10 | 2/10 | 0/10 |
| 100 N | 0.57 m/s | 0/10 | 5/10 | 6/10 |
| 200 N | 1.14 m/s | 0/10 | 10/10 | 10/10 |
| 300 N | 1.71 m/s | 9/10 | 10/10 | 10/10 |
| 400 N | 2.28 m/s | 10/10 | 10/10 | 10/10 |

On flat ground the policy is solid up to 200 N and fails almost completely
at 300 N. That threshold sits just above the 1.0 m/s velocity kick
`push_robot` applied during training, which is the result you would hope
for: it holds everything it was shown and a little more.

The interesting number is the combination. Terrain alone is nearly free and
a 100 N shove on flat is entirely free, but the same shove on rocks or boxes
takes it down half the time. Standing on uneven ground the policy has little
push margin left, and it loses that margin below the disturbance it trained
against.

Part of that gap is the disturbance itself. `push_by_setting_velocity`
overwrites the base velocity, which leaves the feet where they are; a force
applied while a foot is loaded against an edge is the harder of the two, and
training on the velocity kick does not obviously prepare the policy for it.

### Past the training range

Falls out of 10, no pushes, 0.5 m/s for 20 seconds:

| Difficulty | `rough` | `boxes` | `slope` | `inverted_slope` |
| --- | --- | --- | --- | --- |
| 1.0 | 2/10 | 0/10 | 1/10 | 0/10 |
| 1.5 | 5/10 | 3/10 | 2/10 | 5/10 |
| 2.0 | 10/10 | 4/10 | 8/10 | 10/10 |
| 2.5 | 10/10 | 10/10 | 10/10 | 10/10 |
| 3.0 | 10/10 | 10/10 | 10/10 | 10/10 |

Everything survives a little past its training range and nothing survives
much. `rough` is the first to go, gone by twice its trained rock height.
`boxes` extrapolates furthest, still walking on 6 seeds in 10 with steps
twice as tall as anything it saw. The two slopes are the surprise: climbing
is harder than descending, 5/10 against 2/10 at a 30 % grade, which is the
opposite of what the downhill runaway suggests.

### Correlated ground, measured

Falls out of 10, no pushes, 0.5 m/s for 20 seconds:

| Difficulty | Relief | Falls | Median distance |
| --- | --- | --- | --- |
| 1 | 6 cm | 0/10 | 9.61 m |
| 2 | 12 cm | 0/10 | 9.63 m |
| 4 | 24 cm | 0/10 | 9.55 m |
| 6 | 36 cm | 0/10 | 9.47 m |
| 8 | 48 cm | 0/10 | 8.94 m |
| 10 | 60 cm | 2/10 | 8.64 m |

Correlated ground is nearly free. The policy walks over 48 cm of relief
without a fall and pays almost nothing in distance, at eight times the
tallest rock it trained on. Set against `rough` at the same relief the two
are opposite: 12 cm of rock is 10/10 falls, 12 cm of hill is 0/10.

Adding shoves at 24 cm of relief costs that margin anyway:

| Push | flat | `hills` at 24 cm | `rough` at 6 cm |
| --- | --- | --- | --- |
| 0 N | 0/10 | 0/10 | 2/10 |
| 100 N | 0/10 | 3/10 | 5/10 |
| 200 N | 0/10 | 9/10 | 10/10 |

So relief the policy handles perfectly on its own still removes almost all
of its ability to reject a disturbance.

## Reading the numbers

Ten seeds is few. The 95 % interval on 5/10 runs from roughly 0.2 to 0.8, so
neighbouring rows in these tables are mostly not distinguishable and only the
shape of a column is worth reading. Individual runs are chaotic on top of
that: shifting the control phase by a few simulation steps flips a marginal
seed from walking to falling.

This is also sim-to-sim, not ground truth. The policy trained on
`g1_29dof_rev_1_0.urdf` under PhysX and runs here on `unitree_mujoco`'s
MJCF, which differs in contact model and in the foot collision geometry --
four small spheres per sole rather than a mesh. The absolute thresholds
would move on hardware. The ordering between terrains is the part worth
trusting.
