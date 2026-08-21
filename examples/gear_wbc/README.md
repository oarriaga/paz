# GEAR-WBC in MuJoCo

This example runs the ported PAZ GEAR-WBC controller with the Keras JAX
backend. GEAR-WBC is the decoupled whole-body controller used by GR00T N1.5
and N1.6: a command-conditioned reinforcement learning policy that drives the
G1's 12 leg joints and 3 waist joints at 50 Hz. It does not command the arms,
which upstream are handled by a separate inverse-kinematics stack and which
this demo simply holds at their default pose.

Unlike SONIC, GEAR-WBC needs no reference motion. It takes a velocity
command, a base height and a torso orientation, so it is steered directly
from a PlayStation controller. The pad is read outside the MuJoCo viewer,
which leaves the viewer's own key bindings alone instead of toggling
visualization state on every steering input.

The release ships two experts over one architecture. The demo picks between
them exactly as the deployment code does: a velocity command norm at or below
0.05 uses `balance`, anything larger uses `walk`.

## Install

From the PAZ repository:

    python3 -m pip install -e '.[gear_wbc]'

A PlayStation controller is optional. Connect one over USB or Bluetooth to
steer; SDL reads it through `/dev/input/js0`, so no driver setup is needed
beyond pairing it. With no pad connected the robot holds `--speed` forward
for the whole run, which is what makes the terrain measurable.

The demo also needs the G1 MuJoCo plant from a local
GR00T-WholeBodyControl checkout, at

    decoupled_wbc/sim2mujoco/resources/robots/g1

It finds that automatically when the repository is a sibling of PAZ. For
another layout, pass the location explicitly:

    python3 examples/gear_wbc/mujoco_demo.py \
        --scene-dir /path/to/decoupled_wbc/sim2mujoco/resources/robots/g1

`GEAR_WBC_SCENE_DIR` can be set instead of passing the flag every time.

The meshes and policies in that directory are stored with git-lfs. Run
`git lfs pull` in the GR00T-WholeBodyControl checkout first, otherwise
MuJoCo fails to parse the pointer files that stand in for the STL meshes.

## Run

    python3 examples/gear_wbc/mujoco_demo.py

Both experts are compiled with `jax.jit` up front, so the first launch takes
several seconds and neither expert stalls the loop when a command first
selects it. This matters more than it looks: called eagerly, one actor costs
7.7 ms on the CPU and 18.5 ms on a laptop GPU, against a 20 ms control
period, which is enough on its own to drop the demo below real time.

    eager   7.74 ms/call CPU   18.52 ms/call GPU
    jitted  0.27 ms/call CPU    1.15 ms/call GPU

The demo therefore runs on the CPU by default. A single 15-joint actor at
batch one is bound by launch latency rather than throughput, so the CPU wins
per call and leaves the GPU free. To override:

    JAX_PLATFORMS=cuda python3 examples/gear_wbc/mujoco_demo.py

A fixed-length headless run is useful as a smoke check, and as the way to
measure a terrain. It reports how far the robot travelled and how high its
base ended, which is enough to tell walking from falling.

    python3 examples/gear_wbc/mujoco_demo.py --headless --steps 2000 \
        --speed 0.5

`--headless` only drops the viewer. A pad steers the run whenever one is
connected, in both modes, and `--speed` takes over whenever one is not.

## Terrain

`--terrain` picks the ground. The default `flat` is the released scene's
ground plane; `rocky` swaps that plane for a heightfield of rocks up to 3 cm
tall on a 7.8 cm grid, spread over 20 by 20 metres:

    python3 examples/gear_wbc/mujoco_demo.py --terrain rocky

The rocks are uniform noise rescaled by MuJoCo to the heightfield peak, which
is how MuJoCo Playground builds the rough terrain it walks its own G1 over.
`--rock-height` sets that peak in metres, `--seed` redraws the layout, and the
robot spawns one peak higher so it does not start inside a rock:

    python3 examples/gear_wbc/mujoco_demo.py --terrain rocky \
        --rock-height 0.05 --seed 3

GEAR-WBC is blind: the observation carries no terrain, so the policy only
feels the rocks through the base orientation and the joint states. Holding
0.5 m/s forward for 10 s over ten seeds per height, counting a fall below
0.4 m of base height:

| Rock height | Falls | Median distance |
| --- | --- | --- |
| flat | 0/1 | 4.36 m |
| 1 cm | 1/10 | 4.63 m |
| 2 cm | 0/10 | 4.62 m |
| 3 cm | 2/10 | 4.66 m |
| 4 cm | 5/10 | 4.57 m |
| 5 cm | 4/10 | 3.98 m |

Failure is close to binary. A run that stays up covers the same ground it
covers on flat, so the rocks cost almost no speed up to 3 cm, where one seed
in five goes down. At 4 to 5 cm it is roughly a coin flip.

Individual runs are chaotic: shifting the control phase by three simulation
steps flips a marginal seed from walking to falling, so single runs say
little and only the rate over many seeds is worth reading.

## Controls

Every input self-centers, so the whole command is a pure function of the
pad: release it and the robot returns to standing at the default 0.74 m
height with a level torso, back on the balance expert.

| Input | Action |
| --- | --- |
| Left stick | Forward and lateral velocity, up to 0.5 and 0.4 m/s |
| Right stick, left / right | Yaw rate, up to 1.0 rad/s |
| Right stick, up / down | Base height, 0.74 m plus or minus 0.15 m |
| L2 / R2 | Torso roll, up to 20 degrees each way |
| D-pad up / down | Torso pitch |
| D-pad left / right | Torso yaw |

Lateral speed is capped lower than forward speed on purpose: upstream
recommends staying near 0.4 m/s when strafing, because the cross-legged
foot placement of a side step collides at higher speeds.

## Weights

`GearWBC(weights="pretrained")` downloads Keras weights converted from the
released `GR00T-WholeBodyControl-Balance.onnx` and
`GR00T-WholeBodyControl-Walk.onnx`. They are Model Derivatives licensed by
NVIDIA Corporation under the NVIDIA Open Model License.

To convert a local pair of release checkpoints instead:

    python3 -m paz.models.foundation.gear_wbc.conversion \
        --balance_onnx /path/to/GR00T-WholeBodyControl-Balance.onnx \
        --walk_onnx /path/to/GR00T-WholeBodyControl-Walk.onnx
