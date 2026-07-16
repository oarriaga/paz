# FLOWER on LIBERO-OBJECT (MuJoCo)

Closed-loop demonstration of the paz FLOWER VLA port controlling a
Franka Panda in the official LIBERO-OBJECT benchmark, using the official
`mbreuss/flower_libero_object` checkpoint converted to paz weights.

LIBERO, robosuite, and MuJoCo are heavy optional dependencies and are
not part of the paz installation. Install them in a separate virtual
environment (Python 3.10, headless rendering via EGL on an NVIDIA GPU):

```bash
python3 -m venv --system-site-packages libero-venv
./libero-venv/bin/pip install --upgrade pip
git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO
./libero-venv/bin/pip install robosuite==1.4.1 "bddl==1.0.1" easydict \
    "hydra-core==1.2.0" future "gym==0.25.2" cloudpickle
./libero-venv/bin/pip install "mujoco==2.3.7"
./libero-venv/bin/pip install -e ./LIBERO \
    --config-settings editable_mode=compat
echo "N" | ./libero-venv/bin/python -c "import libero.libero"
```

Notes: robosuite 1.4.1 pulls mujoco 3.x, which it cannot use — pin
mujoco 2.3.7. LIBERO needs the compat editable mode because its top
directory is a namespace package. The first import asks for a dataset
path; answering `N` writes `~/.libero/config.yaml` pointing at the
clone. Without EGL (`libEGL_nvidia.so.0`), install `libosmesa6` and use
`MUJOCO_GL=osmesa`.

Run the demo (downloads converted paz weights on first use, or pass a
local directory produced by the converters with `--checkpoint`):

```bash
MUJOCO_GL=egl KERAS_BACKEND=jax ./libero-venv/bin/python \
    examples/flower/demo_libero_mujoco.py \
    --task-index 0 --initial-state-index 0 --seed 0 \
    --output-video flower_libero_rollout.mp4
```

The script loads the LIBERO-OBJECT task, reads its language
instruction, renders the agent-view and wrist cameras, predicts
10-action chunks with the paz FLOWER model (4 rectified-flow Euler
steps), executes them closed loop, saves an MP4 with the instruction
and final status overlaid, and exits nonzero if LIBERO does not report
success. All actions come from the FLOWER policy; there is no scripted
fallback.
