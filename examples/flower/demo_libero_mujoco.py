import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("KERAS_BACKEND", "jax")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from paz.applications import PredictFlowerActions


def build_environment(task, image_size):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bddl_files = get_libero_path("bddl_files")
    bddl_file = os.path.join(bddl_files, task.problem_folder, task.bddl_file)
    kwargs = {"bddl_file_name": bddl_file, "camera_heights": image_size,
              "camera_widths": image_size}
    return OffScreenRenderEnv(**kwargs)


def settle_physics(env, num_steps=5):
    obs = None
    for _ in range(num_steps):
        obs, reward, done, info = env.step(np.zeros(7))
    return obs


def annotate_frame(frame, instruction, success):
    frame = np.ascontiguousarray(frame[:, :, ::-1])
    status = "SUCCESS" if success else "running"
    text = f"{instruction} [{status}]"
    position = (4, frame.shape[0] - 6)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, position, font, 0.32, (255, 255, 255), 1)
    return frame


def save_video(frames, path, fps=20):
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FLOWER on LIBERO-OBJECT")
    add = parser.add_argument
    add("--checkpoint", default=None, help="local converted weights dir")
    add("--task-index", default=0, type=int)
    add("--initial-state-index", default=0, type=int)
    add("--seed", default=0, type=int)
    add("--num-flow-steps", default=4, type=int)
    add("--replan-interval", default=10, type=int)
    add("--max-steps", default=600, type=int)
    add("--image-size", default=256, type=int)
    add("--output-video", default="flower_libero_rollout.mp4")
    args = parser.parse_args()

    from libero.libero import benchmark
    task_suite = benchmark.get_benchmark_dict()["libero_object"]()
    task = task_suite.get_task(args.task_index)
    instruction = task.language
    print(f"task: {task.name}")
    print(f"instruction: {instruction}")

    policy = PredictFlowerActions(
        models_path=args.checkpoint,
        num_flow_steps=args.num_flow_steps,
        seed=args.seed)

    env = build_environment(task, args.image_size)
    env.seed(args.seed)
    env.reset()
    init_states = task_suite.get_task_init_states(args.task_index)
    env.set_init_state(init_states[args.initial_state_index])
    obs = settle_physics(env)

    frames, success, num_steps = [], False, 0
    while num_steps < args.max_steps and not success:
        static_image = obs["agentview_image"]
        wrist_image = obs["robot0_eye_in_hand_image"]
        action_chunk = policy(static_image, wrist_image, instruction)
        for action in action_chunk[:args.replan_interval]:
            obs, reward, done, info = env.step(np.asarray(action))
            num_steps = num_steps + 1
            success = success or bool(done)
            frame = annotate_frame(obs["agentview_image"], instruction,
                                   success)
            frames.append(frame)
            if success or num_steps >= args.max_steps:
                break

    env.close()
    save_video(frames, args.output_video)
    print(f"seed: {args.seed}")
    print(f"steps: {num_steps}")
    print(f"success: {success}")
    print(f"video: {args.output_video}")
    sys.exit(0 if success else 1)
