# SONIC in MuJoCo

This example runs the ported PAZ SONIC actor directly with the Keras JAX
backend. It imports the released ONNX parameters into the PAZ architecture,
then closes the loop around the original 43-DoF G1 MuJoCo plant: 29 SONIC
body joints plus the two 7-DoF Dex3 hands.

The demo intentionally uses the deployment observation layout and controller
constants. It does not call the older Python Keras model or ONNX Runtime for
inference.

Startup matches the deployment workflow: three seconds of pose
initialization, then two seconds of paused batch-1 policy ticks to build
history. The original elastic band is released and playback starts
automatically.

## Install

From the PAZ repository:

    python3 -m pip install -e '.[sonic]'

The demo also needs a local GR00T-WholeBodyControl repository. It reads the
release policy and motions from gear_sonic_deploy and the original 43-DoF
MuJoCo plant from gear_sonic. It automatically finds that repository when it
is a sibling of PAZ, as in the current worktrees.

For another layout, pass the location explicitly:

    python3 examples/sonic/mujoco_demo.py \
        --sonic-root /path/to/gear_sonic_deploy

SONIC_DEPLOY_DIR can be set instead of passing the flag every time.

## Run

Launch the GUI in reference-motion mode:

    python3 examples/sonic/mujoco_demo.py

Select a motion by a case-insensitive part of its name:

    python3 examples/sonic/mujoco_demo.py --motion macarena

Start in virtual 3-point teleoperation mode:

    python3 examples/sonic/mujoco_demo.py \
        --mode teleop --motion walking

The first launch compiles the actor with JAX and can take several seconds.
Only run one copy on a memory-constrained GPU. CPU execution is available with:

    JAX_PLATFORM_NAME=cpu python3 examples/sonic/mujoco_demo.py

## Controls

| Key | Action |
| --- | --- |
| Space | Pause or resume the reference stream |
| Left bracket / right bracket | Previous or next bundled motion |
| Comma / period | Step one frame while paused |
| 1 | Full G1 reference-motion encoder |
| 2 | 3-point teleoperation encoder |
| 3 | SMPL human-motion encoder, when the clip provides SMPL data |
| G | Toggle the original simulator elastic band |
| V | Hide or show reference targets |
| R | Reset the current motion and controller history |
| H | Print the controls in the terminal |
| Esc | Close MuJoCo |

The upper-left HUD shows the PAZ backend, encoder mode, motion, frame,
controller phase, playback state, and elastic-band state. Cyan points show
G1 body targets. Blue, red, and yellow points show the left hand, right hand,
and head targets in teleoperation mode. SMPL joints are magenta.

## Supported SONIC inputs

The released actor has three encoder branches:

- G1 mode consumes ten future full-body joint states and anchor orientations.
  It works with every bundled example clip.
- Teleoperation mode consumes future lower-body states plus the two wrist and
  head targets. Without a headset, this demo follows the original controller's
  fallback and derives those three targets from the bundled motion. This makes
  the branch reproducible and easy to inspect.
- SMPL mode consumes ten frames of 24 root-local human joints, root
  orientations, and G1 wrist joints. The bundled release has no SMPL sample,
  so the GUI enables this branch only when the selected motion directory has a
  smpl_joints.csv file. It must contain a header and one 72-value row per
  motion frame in the release's 24-joint ordering. Missing human joints are
  never replaced by synthetic zeros.

The elastic band is active only during initialization and policy warmup. The
demo releases it before the first playback tick, so the default rollout is
unassisted. Press G to restore it for inspection or recovery. Press R to reset
the physics, controller history, and startup workflow.

## Headless validation

The same entry point provides a deterministic smoke test without a window:

    python3 examples/sonic/mujoco_demo.py \
        --headless --steps 2600 --mode g1 --motion squat

    python3 examples/sonic/mujoco_demo.py \
        --headless --steps 2000 --mode teleop --motion walking

The focused observation and controller tests are:

    pytest -q examples/sonic/simulation_test.py

## Tick-by-tick parity

The comparator consumes logs written by the original C++ controller. Enable
both its split state logs and headerless policy-input log:

    ./target/release/g1_deploy_onnx_ref lo \
        policy/release/model_decoder.onnx reference/example \
        --obs-config policy/release/observation_config.yaml \
        --encoder-file policy/release/model_encoder.onnx \
        --disable-crc-check --enable-csv-logs \
        --logs-dir /tmp/sonic-cpp/state \
        --policy-input-logfile /tmp/sonic-cpp/policy_input.csv

After capturing one motion, replay every logged tick through PAZ:

    python3 examples/sonic/compare_controller_log.py \
        --logs-dir /tmp/sonic-cpp/state \
        --policy-input-logfile /tmp/sonic-cpp/policy_input.csv

The script independently checks the 930-value policy history, the 64-value
encoder token, and the next 29-value action. It preserves the deployment
batch size of one because batching can change an FSQ code at a rounding
boundary.
