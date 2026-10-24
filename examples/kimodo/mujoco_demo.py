"""Generate a G1 motion from text with Kimodo and track it with SONIC.

The prompt is embedded with the ported LLM2Vec encoder, Kimodo samples a
30 Hz motion, the motion is exported to MuJoCo ``qpos`` rows, resampled
to SONIC's 50 Hz clip format, and the existing SONIC MuJoCo demo is
launched on it. Everything before MuJoCo runs in Keras/JAX.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("KERAS_BACKEND", "jax")
# Leave GPU memory for the SONIC subprocess.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jp
import numpy as np

from paz.models.foundation.kimodo import conditioning
from paz.models.foundation.kimodo import features
from paz.models.foundation.kimodo import mujoco as export
from paz.models.foundation.kimodo import sampling
from paz.models.foundation.kimodo.pretrained import KimodoG1RP
from paz.models.foundation.kimodo.text import sanitize_text
from paz.models.foundation.llm2vec.pretrained import LLM2VecLlama3
from paz.models.foundation.llm2vec.pretrained import embed

from sonic_clip import KIMODO_FPS, SONIC_FPS, resample_qpos, write_clip

SONIC_DEMO = Path(__file__).resolve().parents[1] / "sonic" / "mujoco_demo.py"


def embed_prompt(prompt, encoder_path):
    """Sanitizes and embeds the prompt, then frees the 16 GB encoder."""
    llm2vec = LLM2VecLlama3(encoder_path)
    embedding = np.asarray(embed(llm2vec, sanitize_text(prompt)))
    del llm2vec
    return embedding[None]


def build_condition(num_frames, waypoints_path, stats):
    """Root-path constraint from a JSON list of ``[frame, x, z]`` rows."""
    condition = conditioning.build_empty(num_frames)
    if waypoints_path is not None:
        frames, positions_xz = read_waypoints(waypoints_path, num_frames)
        condition = conditioning.add_root_path(condition, frames, positions_xz)
    return conditioning.normalize(condition, stats)


def read_waypoints(path, num_frames):
    waypoints = np.asarray(json.loads(path.read_text()), dtype=np.float32)
    frames = waypoints[:, 0].astype(np.int32)
    if frames.min() < 0 or frames.max() >= num_frames:
        raise ValueError(f"Waypoint frames must lie in [0, {num_frames})")
    return jp.asarray(frames), jp.asarray(waypoints[:, 1:3])


def generate_motion(kimodo, embedding, condition, config, seed):
    text = jp.asarray(embedding, dtype=jp.float32)[None]
    key = jax.random.PRNGKey(seed)
    args = (kimodo.denoiser, text, condition.observed[None],
            condition.mask[None], config)
    sampled = sampling.sample(key, *args)
    return features.decode(sampled[0], kimodo.stats)


def to_qpos(motion):
    to_qpos_frame = jax.vmap(export.to_qpos)
    qpos = to_qpos_frame(motion.local_rotations, motion.root_positions)
    return np.asarray(qpos)


def save_motion(path, motion):
    heading = np.asarray(motion.heading)
    np.savez(
        path,
        posed_joints=np.asarray(motion.joint_positions),
        global_rot_mats=np.asarray(motion.global_rotations),
        local_rot_mats=np.asarray(motion.local_rotations),
        root_positions=np.asarray(motion.root_positions),
        smooth_root_pos=np.asarray(motion.smooth_root_positions),
        foot_contacts=np.asarray(motion.foot_contacts),
        global_root_heading=heading,
    )


def build_sonic_command(clip_dir, headless, video):
    command = [sys.executable, str(SONIC_DEMO), "--motion-dir", str(clip_dir),
               "--motion", "generated", "--mode", "g1"]
    if headless:
        command.append("--headless")
    if video is not None:
        command.extend(["--video", str(video)])
    return command


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default="A person walks forward.")
    parser.add_argument("--models-path", type=Path,
                        help="folder with converted Kimodo assets")
    parser.add_argument("--encoder-path", type=Path,
                        help="folder with converted LLM2Vec assets")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="motion length in seconds")
    parser.add_argument("--num-steps", type=int, default=100,
                        help="DDIM denoising steps")
    parser.add_argument("--text-weight", type=float, default=2.0)
    parser.add_argument("--constraint-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--waypoints", type=Path,
                        help="JSON list of [frame, x, z] root waypoints in "
                             "Kimodo coordinates (meters, z forward)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("kimodo_output"))
    parser.add_argument("--headless", action="store_true",
                        help="run SONIC without a viewer through the clip")
    parser.add_argument("--video", type=Path,
                        help="record the SONIC playback to this mp4")
    args = parser.parse_args()

    print(f"Embedding {sanitize_text(args.prompt)!r} with LLM2Vec")
    embedding = embed_prompt(args.prompt, args.encoder_path)
    print("Loading the pretrained Kimodo-G1-RP-v1 denoiser")
    kimodo = KimodoG1RP(models_path=args.models_path)
    num_frames = int(round(args.duration * KIMODO_FPS))
    config = sampling.SamplingConfig(num_frames, args.num_steps,
                                     args.text_weight, args.constraint_weight,
                                     0.0)
    print(f"Generating {num_frames} frames for {sanitize_text(args.prompt)!r}")
    condition = build_condition(num_frames, args.waypoints, kimodo.stats)
    motion = generate_motion(kimodo, embedding, condition, config, args.seed)
    qpos = to_qpos(motion)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_motion(args.output_dir / "motion.npz", motion)
    np.savetxt(args.output_dir / "qpos.csv", qpos, delimiter=",")
    qpos_sonic = resample_qpos(qpos, KIMODO_FPS, SONIC_FPS)
    clip_dir = args.output_dir / "clips"
    write_clip(clip_dir / "generated", qpos_sonic, SONIC_FPS)
    print(f"Wrote {len(qpos_sonic)} SONIC frames to {clip_dir / 'generated'}")

    command = build_sonic_command(clip_dir, args.headless, args.video)
    print("Launching SONIC:", " ".join(command))
    environment = dict(os.environ)
    if args.video is not None:
        environment.setdefault("MUJOCO_GL", "egl")
    subprocess.run(command, check=True, env=environment)
