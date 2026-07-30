#!/usr/bin/env bash
# Clone PAZ (paz-jax branch) into a fresh virtualenv and run SONIC out of
# the box: no MuJoCo scene, motion clips, or GR00T-WholeBodyControl checkout
# required. Safe to re-run; it reuses an existing checkout and venv.
#
# Usage: ./install.sh [target_dir]
set -euo pipefail

PAZ_REPO="https://github.com/oarriaga/paz.git"
PAZ_DIR="$(realpath "${1:-paz}")"
VENV_DIR="$PAZ_DIR/.venv-sonic"

if [ -d "$PAZ_DIR/.git" ]; then
    echo "Reusing existing checkout at $PAZ_DIR"
    git -C "$PAZ_DIR" fetch origin paz-jax
    git -C "$PAZ_DIR" checkout paz-jax
    git -C "$PAZ_DIR" merge --ff-only origin/paz-jax
else
    git clone --branch paz-jax "$PAZ_REPO" "$PAZ_DIR"
fi

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python3 -m pip install --upgrade pip
python3 -m pip install -e "$PAZ_DIR[sonic]"

echo "Running SONIC (downloads pretrained weights on first run)..."
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
python3 "$PAZ_DIR/examples/sonic/quickstart.py"

cat <<EOF

SONIC is installed at $PAZ_DIR and runs standalone.

To use it again:
    source $VENV_DIR/bin/activate
    python3 $PAZ_DIR/examples/sonic/quickstart.py

The interactive MuJoCo demo also runs out of the box now (it downloads its
G1 scene and reference motions on first use, same as the actor weights):
    python3 $PAZ_DIR/examples/sonic/mujoco_demo.py

See $PAZ_DIR/examples/sonic/README.md for demo controls and modes.
EOF
