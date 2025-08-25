import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"

import jax
import jax.numpy as jp
from jax.random import PRNGKey
import tensorflow_probability.substrates.jax as tfp
from typing import NamedTuple
import matplotlib.pyplot as plt

# --- Assumed Dependencies ---
# pip install git+https://github.com/oarriaga/paz.git
from paz import SE3

# For type hinting
import chex

# Let's use the renderer components from your original script
from paz.backend.graphics import PointLight, Material, Shape, Pattern
from paz.backend.graphics.camera import build_rays
from paz.backend.graphics.render import Render
from paz.backend.graphics import SPHERE, CUBE, PLANE, NO_PATTERN

# --- Constants and Data Structures ---
MAX_NODES = 64
MAX_CHILDREN = 6
NUM_SHAPE_TYPES = 3  # CUBE, SPHERE, PLANE
tfd = tfp.distributions


# --- Helper functions ---
def get_scaling(transform: chex.Array) -> chex.Array:
    """Extracts scaling factors from a 4x4 transformation matrix."""
    return jp.linalg.norm(transform[:3, :3], axis=0)


def rotation_from_axis_angle(axis: chex.Array, angle: float) -> chex.Array:
    """Creates a 4x4 rotation matrix from an axis and an angle."""
    c = jp.cos(angle)
    s = jp.sin(angle)
    t = 1 - c
    x, y, z = axis

    rotation_matrix_3x3 = jp.array(
        [
            [t * x * x + c, t * x * y - z * s, t * x * z + y * s],
            [t * x * y + z * s, t * y * y + c, t * y * z - x * s],
            [t * x * z - y * s, t * y * z + x * s, t * z * z + c],
        ]
    )

    # Embed in a 4x4 homogeneous matrix
    return jp.eye(4).at[:3, :3].set(rotation_matrix_3x3)


class PartNode(NamedTuple):
    is_active: chex.ArrayBatched
    parent_index: chex.ArrayBatched
    shape_type: chex.ArrayBatched
    relative_transform: chex.ArrayBatched
    material: Material


class MetaPriors(NamedTuple):
    """A pytree to hold the priors for a specific scene type."""

    root_shape: int
    root_scale: float
    child_scale_loc: float
    child_scale_scale: float
    child_scale_low: float
    child_scale_high: float
    camera_eye: chex.Array
    light_position: chex.Array
    child_rotation_concentration: float
    # --- NEW: Probabilities for sampling child shape types ---
    shape_probabilities: chex.Array


def create_empty_tree() -> PartNode:
    """Creates a default, inactive tree structure."""
    default_material = Material(
        color=jp.ones(3) * 0.8,
        ambient=0.1,
        diffuse=0.8,
        specular=0.8,
        shininess=100.0,
    )
    material_pytree = jax.tree.map(
        lambda x: jp.broadcast_to(x, (MAX_NODES,) + jp.shape(x)),
        default_material,
    )
    return PartNode(
        is_active=jp.zeros(MAX_NODES, dtype=jp.bool_),
        parent_index=jp.full(MAX_NODES, -1, dtype=jp.int32),
        shape_type=jp.zeros(MAX_NODES, dtype=jp.int32),
        relative_transform=jp.array([jp.eye(4)] * MAX_NODES),
        material=material_pytree,
    )


# --- Prior Sampling Function ---
def sample_from_prior(key: PRNGKey, priors: MetaPriors) -> PartNode:
    """Samples a complete scene tree from the provided meta-priors."""
    key, root_key = jax.random.split(key)
    scene_tree = create_empty_tree()

    root_transform = SE3.scaling(jp.array([priors.root_scale] * 3))
    root_color_key, key = jax.random.split(key)
    root_color = jax.random.uniform(root_color_key, (3,))

    updated_material = scene_tree.material._replace(
        color=scene_tree.material.color.at[0].set(root_color)
    )

    root_node_updates = {
        "is_active": scene_tree.is_active.at[0].set(True),
        "shape_type": scene_tree.shape_type.at[0].set(priors.root_shape),
        "relative_transform": scene_tree.relative_transform.at[0].set(
            root_transform
        ),
        "material": updated_material,
    }
    scene_tree = scene_tree._replace(**root_node_updates)

    def generation_step(carry, parent_idx):
        key, tree = carry
        parent_node = jax.tree.map(lambda x: x[parent_idx], tree)

        def generate_children(key, tree):
            num_active = jp.sum(tree.is_active)

            def child_loop_body(carry, _):
                key, tree, current_slot_idx = carry
                key, activate_key, prop_key, color_key = jax.random.split(
                    key, 4
                )

                activation_prob = 0.8 * (0.9**current_slot_idx)
                activate_child = tfd.Bernoulli(probs=activation_prob).sample(
                    seed=activate_key
                )

                def update_fn(tree):
                    child_idx = current_slot_idx
                    # --- FIX: Need an extra key for shape sampling ---
                    shape_key, scale_key, trans_key, rot_key = jax.random.split(
                        prop_key, 4
                    )

                    # --- NEW: Sample shape type from categorical distribution ---
                    shape_dist = tfd.Categorical(
                        probs=priors.shape_probabilities
                    )
                    child_shape_type = shape_dist.sample(seed=shape_key)

                    scale_dist = tfd.TruncatedNormal(
                        loc=priors.child_scale_loc,
                        scale=priors.child_scale_scale,
                        low=priors.child_scale_low,
                        high=priors.child_scale_high,
                    )
                    scale_factor = scale_dist.sample(seed=scale_key)

                    parent_scale = get_scaling(parent_node.relative_transform)
                    child_scale = parent_scale[0] * scale_factor

                    face_idx = jax.random.randint(trans_key, (), 0, 6)
                    axis_idx = face_idx % 3
                    direction = (face_idx // 3) * 2 - 1
                    offset = 1.2
                    translation = (
                        jp.zeros(3)
                        .at[axis_idx]
                        .set(
                            direction
                            * (parent_scale[axis_idx] + child_scale)
                            / 2
                            * offset
                        )
                    )

                    angle = tfd.VonMises(
                        0.0, priors.child_rotation_concentration
                    ).sample(seed=rot_key)
                    axis = jax.random.normal(rot_key, (3,))
                    axis /= jp.linalg.norm(axis)
                    rotation_matrix = rotation_from_axis_angle(axis, angle)

                    child_transform = (
                        SE3.translation(translation)
                        @ rotation_matrix
                        @ SE3.scaling(jp.full(3, child_scale))
                    )

                    parent_color = parent_node.material.color
                    color_noise = tfd.Normal(0.0, 0.1).sample(3, seed=color_key)
                    child_color = jp.clip(parent_color + color_noise, 0.0, 1.0)

                    updated_material = tree.material._replace(
                        color=tree.material.color.at[child_idx].set(child_color)
                    )

                    updates = {
                        "is_active": tree.is_active.at[child_idx].set(True),
                        "parent_index": tree.parent_index.at[child_idx].set(
                            parent_idx
                        ),
                        "shape_type": tree.shape_type.at[child_idx].set(
                            child_shape_type
                        ),
                        "relative_transform": tree.relative_transform.at[
                            child_idx
                        ].set(child_transform),
                        "material": updated_material,
                    }
                    return tree._replace(**updates)

                has_space = current_slot_idx < MAX_NODES
                tree = jax.lax.cond(
                    activate_child & has_space, update_fn, lambda t: t, tree
                )
                return (key, tree, jp.sum(tree.is_active)), None

            (key, tree, _), _ = jax.lax.scan(
                child_loop_body,
                (key, tree, num_active),
                jp.arange(MAX_CHILDREN),
            )
            return key, tree

        key, tree = jax.lax.cond(
            parent_node.is_active,
            generate_children,
            lambda k, t: (k, t),
            key,
            tree,
        )
        return (key, tree), None

    (final_key, final_tree), _ = jax.lax.scan(
        generation_step, (key, scene_tree), jp.arange(MAX_NODES)
    )
    return final_tree


# --- Tree to Scene Compiler ---
@jax.jit
def compile_tree_to_scene(scene_tree: PartNode):
    world_transforms = scene_tree.relative_transform

    def body(i, transforms):
        node = jax.tree.map(lambda x: x[i], scene_tree)
        parent_transform = transforms[node.parent_index]
        current_world_transform = parent_transform @ node.relative_transform

        return jax.lax.cond(
            node.parent_index >= 0,
            lambda t: t.at[i].set(current_world_transform),
            lambda t: t,
            transforms,
        )

    world_transforms = jax.lax.fori_loop(1, MAX_NODES, body, world_transforms)

    dummy_pattern = Pattern(
        transform=jp.array([jp.eye(4)] * MAX_NODES),
        type=jp.full((MAX_NODES,), NO_PATTERN, dtype=jp.int32),
        image=jp.zeros((MAX_NODES, 1, 1, 3)),
    )

    scene_shapes = Shape(
        transform=world_transforms,
        type=scene_tree.shape_type,
        pattern=dummy_pattern,
        material=scene_tree.material,
    )
    mask = scene_tree.is_active
    return scene_shapes, mask


# --- Main Visualization Script ---
def main():
    H, W = 240, 320
    key = PRNGKey(42)

    # 1. Define the Meta-Priors for our two scene types
    object_priors = MetaPriors(
        root_shape=CUBE,
        root_scale=1.0,
        child_scale_loc=0.6,
        child_scale_scale=0.2,
        child_scale_low=0.2,
        child_scale_high=0.9,
        camera_eye=jp.array([0.0, 0.0, -8.0]),
        light_position=jp.array([-10.0, 10.0, -10.0]),
        child_rotation_concentration=8.0,
        # --- NEW: Sample cubes and spheres for objects ---
        shape_probabilities=jp.array(
            [0.6, 0.4, 0.0]
        ),  # 60% CUBE, 40% SPHERE, 0% PLANE
    )
    room_priors = MetaPriors(
        root_shape=CUBE,
        root_scale=2.5,
        child_scale_loc=1.0,
        child_scale_scale=0.1,
        child_scale_low=0.95,
        child_scale_high=1.05,
        camera_eye=jp.array([0.0, 0.0, -2.4]),
        light_position=jp.array([0.0, 2.0, 0.0]),
        child_rotation_concentration=100.0,
        # --- NEW: Sample mostly cubes for rooms ---
        shape_probabilities=jp.array(
            [0.9, 0.05, 0.05]
        ),  # 90% CUBE, 5% SPHERE, 5% PLANE
    )

    # 2. Setup Plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    fig.suptitle("Samples from Prior Predictive Distribution", fontsize=16)

    # 3. Sample, Compile, Render, and Visualize
    for i in range(4):
        print(f"Generating scene {i+1}/4...")
        key, sample_key, choice_key = jax.random.split(key, 3)

        choice = jax.random.randint(choice_key, (), 0, 2)
        priors = jax.lax.cond(
            choice == 0, lambda: object_priors, lambda: room_priors
        )

        if choice == 0:
            scene_type_str = "object"
        else:
            scene_type_str = "room"

        # 4. Setup Renderer with scene-specific priors
        lights = [PointLight(jp.array([1.0, 1.0, 1.0]), priors.light_position)]
        camera_pose = SE3.view_transform(
            priors.camera_eye,
            jp.array([0.0, 0.0, 0.0]),
            jp.array([0.0, 1.0, 0.0]),
        )
        rays = build_rays((H, W), jp.pi / 4, camera_pose)
        render = Render((H, W), camera_pose, rays, shadows=False)
        fast_render = jax.jit(render)

        scene_tree = jax.jit(sample_from_prior)(sample_key, priors)
        scene_shapes, mask = compile_tree_to_scene(scene_tree)
        image, depth = fast_render(scene_shapes, mask, lights)

        ax = axes[i]
        ax.imshow(jp.clip(image, 0.0, 1.0))
        ax.set_title(f"Sample {i+1}: type='{scene_type_str}'")
        ax.axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main()
