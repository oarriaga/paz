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

## Encoding new prompts

Text goes through Kimodo's official LLM2Vec encoder (Llama-3-8B-Instruct
with two LoRA adapters), which is a torch model and is kept out of the JAX
path. Encode prompts once and pass the file to the demo:

    PYTHONPATH=/path/to/nv-tlabs/kimodo python3 examples/kimodo/encode_text.py \
        "A person waves." "A person sits down." --output my_prompts.npz
    python3 examples/kimodo/mujoco_demo.py "A person waves." \
        --prompts my_prompts.npz

`encode_text.py` needs torch, `transformers==5.1`, `peft`, and a clone of
the upstream repository on `PYTHONPATH`. It loads the Llama base from the
ungated `NousResearch/Meta-Llama-3-8B-Instruct` mirror after checking that
its shards carry the sha256 of the official gated files, so no Hugging
Face access request is needed. Encoding takes about 16 GB of GPU memory in
bfloat16; use `--device cpu` on smaller cards.

## Tests

    pytest paz/models/foundation/kimodo examples/kimodo

Parity against the torch reference and the end-to-end simulation are
opt-in:

    KIMODO_FIXTURES=~/.keras/paz/models/kimodo/reference_fixtures.npz \
    KIMODO_WEIGHTS_DIR=~/.keras/paz/models/kimodo \
    pytest paz/models/foundation/kimodo/parity_test.py -s

    KIMODO_PIPELINE_TEST=1 pytest examples/kimodo/pipeline_test.py -s

`reference.py` captures the fixtures from the upstream implementation and
`conversion.py` converts the released safetensors checkpoint; both are
one-time development tools that need torch.
