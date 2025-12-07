import os
import json
import time
import subprocess
import pytest
import jax
import jax.numpy as jp

# from termcolor import colored  # Optional: makes output easier to read
from paz import SE3
from paz.graphics import renderer
from paz.graphics import (
    PointLight,
    Sphere,
    Cube,
    Plane,
    Cylinder,
    Cone,
    Material,
    Scene,
    SphericalPattern,
    PlanarPattern,
    CylindricalPattern,
)

BENCHMARK_FILE = "benchmarks.json"
RESOLUTIONS = [(128, 128), (256, 256)]
N_REPEATS = 20  # Number of times to run the render to calculate the mean

# --- UTILITIES ---


def get_git_commit():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")
        )
    except:
        return "unknown_commit"


def load_benchmarks():
    if os.path.exists(BENCHMARK_FILE):
        with open(BENCHMARK_FILE, "r") as f:
            return json.load(f)
    return None


def save_benchmarks(data):
    with open(BENCHMARK_FILE, "w") as f:
        json.dump(data, f, indent=4)


# --- SCENE SETUP ---

GREEN = (85 / 255, 181 / 255, 103 / 255)
WHITE = (1.0, 1.0, 1.0)


def CheckeredImage(box_size=50, rows=8, cols=8, color_A=GREEN, color_B=WHITE):
    checkered = jp.indices((rows, cols)).sum(axis=0) % 2
    image_channels = []
    for channel_arg in range(3):
        checkered_channel = jp.kron(checkered, jp.ones((box_size, box_size)))
        checkered_color_A = color_A[channel_arg] * checkered_channel
        checkered_color_B = color_B[channel_arg] * (1 - checkered_channel)
        checkered_channel = checkered_color_A + checkered_color_B
        image_channels.append(jp.expand_dims(checkered_channel, axis=-1))
    return jp.concatenate(image_channels, axis=-1)


@pytest.fixture
def simple_scene_data():
    camera_pose = SE3.view_transform(
        jp.array([0.0, -2.0, 4.0]),
        jp.array([0.0, 0.0, 0.0]),
        jp.array([0.0, 1.0, 0.0]),
    )
    lights = [PointLight(jp.full(3, 1.2), jp.array([3.0, 3.0, 5.0]))]
    material = Material(
        color=jp.array([1.0, 0.0, 0.0]),
        ambient=0.3,
        diffuse=0.5,
        specular=0.8,
        shininess=16.0,
    )
    scene = Scene([Sphere(jp.eye(4), material)])
    return scene, lights, camera_pose


@pytest.fixture
def complex_scene_data():
    camera_pose = SE3.view_transform(
        jp.array([0.0, 8.0, 8.0]),
        jp.array([0.0, 0.0, 0.0]),
        jp.array([0.0, 1.0, 0.0]),
    )
    lights = [
        PointLight(jp.ones(3) / 5, jp.array([5.0, 4.0, 8.0])),
        PointLight(jp.ones(3) / 5, jp.array([5.0, 4.0, -8.0])),
    ]
    checkered = CheckeredImage()
    mat = Material(jp.zeros(3), 0.3, 0.1, 0.0, 100.0)
    scene = Scene(
        [
            Sphere(
                SE3.translation(jp.array([0.0, 1.0, -3.0])),
                mat,
                SphericalPattern(checkered),
            ),
            Cylinder(
                SE3.translation(jp.array([0.0, 1.0, -1.0]))
                @ SE3.scaling(jp.full(3, 1.0)),
                mat,
                CylindricalPattern(checkered, SE3.scaling(jp.full(3, 3.0))),
            ),
            Cone(
                SE3.translation(jp.array([0.0, 1.0, 1.0])),
                mat,
                PlanarPattern(checkered, SE3.scaling(jp.full(3, 3.0))),
            ),
            Cube(
                SE3.translation(jp.array([0.0, 1.0, 3.0]))
                @ SE3.scaling(jp.full(3, 1.0)),
                mat,
                CylindricalPattern(checkered, SE3.scaling(jp.full(3, 4.0))),
            ),
            Plane(),
        ]
    )
    return scene, lights, camera_pose


# --- BENCHMARK EXECUTION ---


@pytest.mark.parametrize("resolution", RESOLUTIONS)
@pytest.mark.parametrize("scene_type", ["simple", "complex"])
@pytest.mark.parametrize("shadows", [True, False])
def test_benchmark_jit_renderer(
    scene_type, shadows, resolution, simple_scene_data, complex_scene_data
):
    # 1. Setup
    if scene_type == "simple":
        scene, lights, camera_pose = simple_scene_data
    else:
        scene, lights, camera_pose = complex_scene_data

    H, W = resolution
    y_FOV = jp.pi / 4.0
    rays = renderer.paz.graphics.camera.build_rays((H, W), y_FOV, camera_pose)

    # 2. JIT Compile
    @jax.jit
    def run_render():
        return renderer.render(
            (H, W), camera_pose, rays, scene, lights, mask=None, shadows=shadows
        )

    # Warmup (Compile time)
    print(
        f"\n--- Warming up {scene_type} {resolution} (Shadows: {shadows}) ---"
    )
    _ = run_render()
    jax.block_until_ready(_)

    # 3. Execution (Timing Loop)
    durations = []
    for i in range(N_REPEATS):
        start_time = time.perf_counter()
        result = run_render()
        jax.block_until_ready(result)
        end_time = time.perf_counter()
        durations.append(end_time - start_time)

    # Calculate Mean
    current_duration = sum(durations) / len(durations)

    # --- LOGIC FOR COMPARISON AND SAVING ---

    current_commit = get_git_commit()
    benchmark_data = load_benchmarks()

    # Create a unique key for this specific test case
    test_key = f"{scene_type}_{H}x{W}_shadows_{shadows}"

    # Initialize data structure if empty
    if benchmark_data is None:
        benchmark_data = {"commit": current_commit, "results": {}}

    saved_commit = benchmark_data.get("commit", "unknown")
    saved_results = benchmark_data.get("results", {})

    print(
        f"\n[Result] {test_key} (Mean of {N_REPEATS}): {current_duration:.5f}s"
    )

    # LOGIC BRANCHES

    # 1. New Benchmark (Entry doesn't exist)
    if test_key not in saved_results:
        print("-> New benchmark detected. Saving...")
        saved_results[test_key] = current_duration
        benchmark_data["results"] = saved_results
        benchmark_data["commit"] = current_commit
        save_benchmarks(benchmark_data)

    # 2. Same Commit (Compare only, treat as uncommited changes check)
    elif saved_commit == current_commit:
        prev_duration = saved_results[test_key]
        diff = prev_duration - current_duration
        pct = (diff / prev_duration) * 100

        if diff > 0:
            print(f"-> Performance: FASTER by {abs(diff):.5f}s ({pct:.2f}%)")
        else:
            print(f"-> Performance: SLOWER by {abs(diff):.5f}s ({pct:.2f}%)")

    # 3. New Commit (Compare and Overwrite ONLY if better)
    else:
        prev_duration = saved_results[test_key]
        diff = prev_duration - current_duration
        pct = (diff / prev_duration) * 100

        if current_duration < prev_duration:
            print(
                f"-> IMPROVEMENT DETECTED (New Commit). Overwriting benchmark."
            )
            print(
                f"-> Old: {prev_duration:.5f}s | New: {current_duration:.5f}s | Imp: {pct:.2f}%"
            )

            # Update only this key and the commit hash
            saved_results[test_key] = current_duration
            benchmark_data["results"] = saved_results
            benchmark_data["commit"] = current_commit
            save_benchmarks(benchmark_data)
        else:
            print(f"-> WARNING: REGRESSION DETECTED in new commit.")
            print(
                f"-> Old: {prev_duration:.5f}s | New: {current_duration:.5f}s | Regression: {pct:.2f}%"
            )
            print("-> Keeping original benchmark file.")
