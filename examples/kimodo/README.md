# Kimodo text-to-motion with SONIC in MuJoCo

This example turns a sentence into a Unitree G1 reference motion with the
ported NVIDIA Kimodo-G1-RP-v1 diffusion model
(`paz.models.foundation.kimodo`) and tracks it with the existing PAZ SONIC
controller in MuJoCo (`examples/sonic/mujoco_demo.py`). Kimodo is a
kinematic motion generator; SONIC stays responsible for the physics.

    "A person walks forward."
        -> LLM2Vec embedding (precomputed)
        -> Kimodo DDIM sampling (Keras/JAX)
        -> 34-joint G1 motion -> MuJoCo qpos -> SONIC 50 Hz clip
        -> SONIC actor in MuJoCo

## Install

From the PAZ repository:

    python3 -m pip install -e '.[sonic]'

The first run downloads the converted Kimodo weights (about 1.1 GB), the
normalization statistics, a small set of prompt embeddings, and the SONIC
actor, scene and motions. Weights follow the NVIDIA Open Model License.

## Run

Generate the default prompt and watch SONIC track it:

    python3 examples/kimodo/mujoco_demo.py

Pick another bundled prompt, a longer motion, or a different sample:

    python3 examples/kimodo/mujoco_demo.py "A person walks in a circle." \
        --duration 6 --seed 3

Headless smoke run (SONIC stops after its startup plus the clip):

    python3 examples/kimodo/mujoco_demo.py --headless

Outputs land in `--output-dir` (default `kimodo_output/`): `motion.npz`
with the reference keys (`posed_joints`, `global_rot_mats`,
`local_rot_mats`, `root_positions`, `smooth_root_pos`, `foot_contacts`,
`global_root_heading`), `qpos.csv` (30 Hz MuJoCo rows), and
`clips/generated/` in the SONIC clip-folder format. The clip folder can be
replayed on its own with:

    python3 examples/sonic/mujoco_demo.py --motion-dir kimodo_output/clips

A root path can be imposed with `--waypoints`, a JSON list of
`[frame, x, z]` rows in Kimodo coordinates (meters, z forward, 30 Hz
frames); the planar smooth-root position is pinned at those frames and
the constraint guidance weight applies:

    echo '[[0, 0, 0], [75, 0, 1.5], [149, 1.5, 1.5]]' > turn.json
    python3 examples/kimodo/mujoco_demo.py --waypoints turn.json

Sampling knobs mirror the reference CLI: `--num-steps` (DDIM steps,
default 100), `--text-weight` and `--constraint-weight` (separated
classifier-free guidance, default 2.0 each). Post-processing is not
applied, matching the reference, which disables it for G1.

## Text encoder

Prompts are embedded natively by the ported LLM2Vec encoder
(`paz.models.foundation.llm2vec`): the bidirectional Llama-3-8B-Instruct
with both LLM2Vec LoRA adapters merged, run one prompt at a time with mean
pooling exactly as the reference does. The first run downloads its 16 GB
bfloat16 weights; the encoder needs about 17 GB of GPU memory and is freed
before SONIC starts. Any sentence works, for example:

    python3 examples/kimodo/mujoco_demo.py "A person waves, then sits down."

## Tests

    pytest paz/models/foundation/kimodo paz/models/foundation/llm2vec \
        examples/kimodo

Parity against the torch references and the end-to-end simulation are
opt-in:

    KIMODO_FIXTURES=~/.keras/paz/models/kimodo/reference_fixtures.npz \
    KIMODO_WEIGHTS_DIR=~/.keras/paz/models/kimodo \
    pytest paz/models/foundation/kimodo/parity_test.py -s

    LLM2VEC_FIXTURES=~/.keras/paz/models/llm2vec_fixtures.npz \
    LLM2VEC_WEIGHTS_DIR=~/.keras/paz/models/llm2vec \
    pytest paz/models/foundation/llm2vec/parity_test.py -s

    KIMODO_PIPELINE_TEST=1 pytest examples/kimodo/pipeline_test.py -s

Each package has a `reference.py` that captures fixtures from the upstream
torch implementation and a `conversion.py` that converts the released
checkpoints (the LLM2Vec one merges the LoRA adapters and verifies the
Llama mirror against the sha256 of the official gated shards); these are
one-time development tools that need torch.
