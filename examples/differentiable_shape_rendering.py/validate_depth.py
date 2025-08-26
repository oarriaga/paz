import matplotlib.pyplot as plt
import trimesh
import pyrender
import jax.numpy as jp
import paz
from paz import SE3

# --- 1. Core Comparison Logic ---


def compare_renderers(scene_name, shape_definitions, camera_pose, camera_name):
    """
    Creates, renders, and compares a scene in both pyrender and paz.
    """
    print(f"--- Running test: {scene_name} from {camera_name} ---")

    # -- A. Create the paz scene --
    paz_shapes = []
    for shape_def in shape_definitions:
        scaling_transform = SE3.scaling(shape_def["paz_scaling"])
        paz_shapes.append(
            paz.graphics.Shape(
                shape_def["pose"] @ scaling_transform,
                shape_def["paz_type"],
                default_pattern,
                default_material,
            )
        )
    paz_scene = paz.graphics.shapes.merge(*paz_shapes)

    # -- B. Create the pyrender scene --
    pyrender_scene = pyrender.Scene()
    for shape_def in shape_definitions:
        mesh = trimesh.creation.__dict__[shape_def["trimesh_shape"]](
            **shape_def["trimesh_args"]
        )
        pyrender_scene.add(
            pyrender.Mesh.from_trimesh(mesh), pose=shape_def["pose"]
        )

    # -- C. Setup cameras for both renderers --
    openGL_to_paz = jp.eye(4)
    paz_world_to_camera = openGL_to_paz @ camera_pose
    pyrender_camera_to_world = jp.linalg.inv(camera_pose)

    pyrender_scene.add(pyrender_camera, pose=pyrender_camera_to_world)
    pyrender_scene.add(light, pose=pyrender_camera_to_world)

    # -- D. Render both scenes --
    pyrender_renderer = pyrender.OffscreenRenderer(W, H)
    true_image, true_depth = pyrender_renderer.render(pyrender_scene)

    paz_rays = paz.graphics.camera.build_rays(
        (H, W), y_FOV, paz_world_to_camera
    )
    pred_image, pred_depth = paz.graphics.render(
        (H, W),
        paz_world_to_camera,
        paz_rays,
        paz_scene,
        jp.ones(len(paz_shapes)),
        lights,
    )

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Scene: '{scene_name}' from Camera: '{camera_name}'", fontsize=16
    )
    axs[0].imshow(true_image)
    axs[0].set_title("Ground Truth (pyrender)")
    axs[1].imshow(pred_image)
    axs[1].set_title("Prediction (paz)")
    diff_im = axs[2].imshow(jp.abs(true_depth - pred_depth))
    axs[2].set_title("Absolute Difference")
    fig.colorbar(diff_im, ax=axs[2])
    plt.show()

    # -- E. Compare and assert --
    assert jp.allclose(
        true_depth, pred_depth, atol=1e-2
    ), f"Depth mismatch for {scene_name} from {camera_name}"
    print("Depth maps match! ✔️")

    # -- F. Visualize the results --


# --- 2. Global Scene Configuration ---

H = 120
W = 160
y_FOV = jp.pi / 3.0

default_pattern = paz.graphics.Pattern(
    jp.eye(4), paz.graphics.NO_PATTERN, jp.ones((1, 1, 3))
)
default_material = paz.graphics.Material(
    jp.full((3,), 0.8), 0.1, 0.9, 0.0, 200.0
)
pyrender_camera = pyrender.PerspectiveCamera(yfov=y_FOV, aspectRatio=W / H)
light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=10.0)

# --- 3. Define Scenes and Camera Poses ---

# FIX: Use a Z-up convention to match the demo script and pyrender's default.
Z_UP = jp.array([0.0, 0.0, 1.0])
camera_poses = {
    "Front View": SE3.view_transform(
        camera_origin=jp.array([0.0, -4.0, 2.0]),
        target_position=jp.zeros(3),
        world_up=Z_UP,
    ),
    "Side View": SE3.view_transform(
        camera_origin=jp.array([-4.0, 0.0, 1.0]),
        target_position=jp.zeros(3),
        world_up=Z_UP,
    ),
    "Top-Down View": SE3.view_transform(
        camera_origin=jp.array([0.0, 0.0, 5.0]),  # High on Z-axis
        target_position=jp.zeros(3),
        world_up=Z_UP,
    ),
}

scenes = {
    "Cube on Plane": [
        {
            "paz_type": paz.graphics.PLANE,
            "paz_scaling": jp.ones(3),
            "trimesh_shape": "box",
            "trimesh_args": {"extents": [10.0, 10.0, 0.01]},
            # FIX: Rotate the paz canonical plane to lie on the XY plane.
            "pose": SE3.rotation_x(jp.pi / 2.0),
        },
        {
            "paz_type": paz.graphics.CUBE,
            "paz_scaling": jp.array([0.5, 0.5, 1.0]),
            "trimesh_shape": "box",
            "trimesh_args": {"extents": [1.0, 1.0, 2.0]},
            # FIX: Place the cube on top of the plane at z=0.
            "pose": SE3.translation(jp.array([0.0, 0.0, 1.0])),
        },
    ],
    "Sphere and Cylinder": [
        {
            "paz_type": paz.graphics.SPHERE,
            "paz_scaling": jp.array([0.8, 0.8, 0.8]),
            "trimesh_shape": "icosphere",
            "trimesh_args": {"radius": 0.8},
            "pose": SE3.translation(jp.array([-1.5, 0.0, 0.8])),
        },
        {
            "paz_type": paz.graphics.CYLINDER,
            "paz_scaling": jp.array([0.5, 0.5, 1.0]),
            "trimesh_shape": "cylinder",
            "trimesh_args": {"radius": 0.5, "height": 2.0},
            "pose": SE3.translation(jp.array([1.5, 0.0, 1.0])),
        },
    ],
}


# --- 4. Main Execution Loop ---

if __name__ == "__main__":
    for camera_name, camera_pose in camera_poses.items():
        lights = [
            paz.graphics.PointLight(jp.full((3,), 3.0), camera_pose[:3, 3])
        ]

        for scene_name, shape_definitions in scenes.items():
            compare_renderers(
                scene_name, shape_definitions, camera_pose, camera_name
            )
