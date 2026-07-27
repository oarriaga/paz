# GEAR-WBC in MuJoCo

This example runs the ported PAZ GEAR-WBC controller with the Keras JAX
backend. GEAR-WBC is the decoupled whole-body controller used by GR00T N1.5
and N1.6: a command-conditioned reinforcement learning policy that drives the
G1's 12 leg joints and 3 waist joints at 50 Hz. It does not command the arms,
which upstream are handled by a separate inverse-kinematics stack and which
this demo simply holds at their default pose.

Unlike SONIC, GEAR-WBC needs no reference motion. It takes a velocity
command, a base height and a torso orientation, so it is steered directly
from the keyboard.

The release ships two experts over one architecture. The demo picks between
them exactly as the deployment code does: a velocity command norm at or below
0.05 uses `balance`, anything larger uses `walk`.

## Install

From the PAZ repository:

    python3 -m pip install -e '.[gear_wbc]'

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

The first launch compiles the actors with JAX and can take several seconds.
CPU execution is available with:

    JAX_PLATFORMS=cpu python3 examples/gear_wbc/mujoco_demo.py

A fixed-length headless run is useful as a smoke check:

    python3 examples/gear_wbc/mujoco_demo.py --headless --steps 1000

## Controls

| Key | Action |
| --- | --- |
| w / s | Increase or decrease forward velocity |
| a / d | Increase or decrease lateral velocity |
| q / e | Increase or decrease yaw rate |
| z | Stop, returning to the balance expert |
| 1 / 2 | Raise or lower the commanded base height |
| 3 / 4 | Torso roll |
| 5 / 6 | Torso pitch |
| 7 / 8 | Torso yaw |

## Weights

`GearWBC(weights="pretrained")` downloads Keras weights converted from the
released `GR00T-WholeBodyControl-Balance.onnx` and
`GR00T-WholeBodyControl-Walk.onnx`. They are Model Derivatives licensed by
NVIDIA Corporation under the NVIDIA Open Model License.

To convert a local pair of release checkpoints instead:

    python3 -m paz.models.foundation.gear_wbc.conversion \
        --balance_onnx /path/to/GR00T-WholeBodyControl-Balance.onnx \
        --walk_onnx /path/to/GR00T-WholeBodyControl-Walk.onnx
