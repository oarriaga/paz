import os
import jax
import jax.numpy as jp
from jax.random import PRNGKey
import tensorflow_probability.substrates.jax as tfp
from typing import NamedTuple
import matplotlib.pyplot as plt
import graphviz
from paz import SE3
import chex

from paz.backend.graphics import PointLight, Material, Shape, Pattern
from paz.backend.graphics.camera import build_rays
from paz.backend.graphics.render import Render
from paz.backend.graphics import SPHERE, CUBE, PLANE, NO_PATTERN

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"

# --- Constants and Data Structures ---
MAX_NODES = 64
MAX_CHILDREN = 6
SHAPE_NAMES = {CUBE: "CUBE", SPHERE: "SPHERE", PLANE: "PLANE"}
tfd = tfp.distributions


class PartNode(NamedTuple):
    is_active: chex.ArrayBatched
    parent_index: chex.ArrayBatched
    shape_type: chex.ArrayBatched
    relative_transform: chex.ArrayBatched
    material: Material


class MetaPriors(NamedTuple):
    root_shape_probabilities: chex.Array
    root_scale: float
    child_scale_loc: float
    child_scale_scale: float
    child_scale_low: float
    child_scale_high: float
    camera_eye: chex.Array
    light_position: chex.Array
    child_rotation_concentration: float
    child_shape_probabilities: chex.Array
    child_activation_prob_base: float
    child_activation_prob_decay: float
    face_probabilities: chex.Array


def get_scaling_from_transform(transform_mat):
    """Extracts scaling factors from a 4x4 transformation matrix."""
    return jp.linalg.norm(transform_mat[:3, :3], axis=0)


def rotation_from_axis_angle(axis, angle):
    """Creates a 4x4 rotation matrix from an axis and an angle."""
    c = jp.cos(angle)
    s = jp.sin(angle)
    t = 1 - c
    x, y, z = axis

    rot_mat_3x3 = jp.array(
        [
            [t * x * x + c, t * x * y - z * s, t * x * z + y * s],
            [t * x * y + z * s, t * y * y + c, t * y * z - x * s],
            [t * x * z - y * s, t * y * z + x * s, t * z * z + c],
        ]
    )
    return jp.eye(4).at[:3, :3].set(rot_mat_3x3)


def transform_points(affine_transform, points):
    """Applies an affine transformation to a set of points."""
    points_hom = jp.concatenate(
        [points, jp.ones((*points.shape[:-1], 1))], axis=-1
    )
    transformed_hom_points = points_hom @ affine_transform.T
    return transformed_hom_points[..., :3]


def transform_vectors(affine_transform, vectors):
    """Applies an affine transformation to a set of vectors."""
    return vectors @ affine_transform[:3, :3].T


def transform_rays(affine_transform, ray_origins, ray_directions):
    """Applies an affine transformation to rays."""
    trans_ray_origins = transform_points(affine_transform, ray_origins)
    trans_ray_directions = transform_vectors(affine_transform, ray_directions)
    return trans_ray_origins, trans_ray_directions


def create_default_material():
    """Creates a default material for the scene."""
    return Material(
        color=jp.ones(3) * 0.8,
        ambient=0.1,
        diffuse=0.8,
        specular=0.8,
        shininess=100.0,
    )


def create_empty_tree():
    """Initializes an empty PartNode tree with default values."""
    default_material = create_default_material()
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


def init_root_node(key, scene_tree, priors):
    """Initializes the root node of the scene tree."""
    shape_key, color_key = jax.random.split(key)

    root_shape_dist = tfd.Categorical(probs=priors.root_shape_probabilities)
    root_shape = root_shape_dist.sample(seed=shape_key)

    root_transform = SE3.scaling(jp.array([priors.root_scale] * 3))
    root_color = jax.random.uniform(color_key, (3,))

    updated_material = scene_tree.material._replace(
        color=scene_tree.material.color.at[0].set(root_color)
    )

    root_node_updates = {
        "is_active": scene_tree.is_active.at[0].set(True),
        "shape_type": scene_tree.shape_type.at[0].set(root_shape),
        "relative_transform": scene_tree.relative_transform.at[0].set(
            root_transform
        ),
        "material": updated_material,
    }
    return scene_tree._replace(**root_node_updates)


def sample_child_shape(key, priors):
    """Samples a shape type for a new child node."""
    shape_dist = tfd.Categorical(probs=priors.child_shape_probabilities)
    return shape_dist.sample(seed=key)


def sample_child_scale(key, priors, parent_transform):
    """Samples a scale for a new child node relative to its parent."""
    scale_dist = tfd.TruncatedNormal(
        loc=priors.child_scale_loc,
        scale=priors.child_scale_scale,
        low=priors.child_scale_low,
        high=priors.child_scale_high,
    )
    scale_factor = scale_dist.sample(seed=key)
    parent_scale_vec = get_scaling_from_transform(parent_transform)
    child_scale = parent_scale_vec[0] * scale_factor
    return child_scale, parent_scale_vec


def calculate_translation_on_face(
    axis_idx, direction, parent_scale_vec, child_scale
):
    """Calculates the translation vector to place a child on a parent's face."""
    offset = 1.05
    # Calculate the distance to move along the selected axis
    distance = (parent_scale_vec[axis_idx] * 0.5 + child_scale * 0.5) * offset
    # Create the translation vector
    translation_vector = jp.zeros(3).at[axis_idx].set(direction * distance)
    return translation_vector


def calculate_child_placement(key, priors, child_scale, parent_scale_vec):
    """Calculates the translation and rotation for a new child node."""
    trans_key, rot_key = jax.random.split(key)

    face_dist = tfd.Categorical(probs=priors.face_probabilities)
    face_idx = face_dist.sample(seed=trans_key)
    axis_idx = face_idx % 3
    direction = (face_idx // 3) * 2 - 1

    # --- REFACTORED: Use the new helper function ---
    translation = calculate_translation_on_face(
        axis_idx, direction, parent_scale_vec, child_scale
    )

    angle = tfd.VonMises(0.0, priors.child_rotation_concentration).sample(
        seed=rot_key
    )
    axis = jax.random.normal(rot_key, (3,))
    axis /= jp.linalg.norm(axis)
    rotation = rotation_from_axis_angle(axis, angle)
    return translation, rotation


def determine_child_color(key, parent_color):
    """Determines a new child's color based on its parent's color."""
    color_noise = tfd.Normal(0.0, 0.1).sample(3, seed=key)
    return jp.clip(parent_color + color_noise, 0.0, 1.0)


def generate_child(carry, child_num):
    """Generates a single child for a parent node if conditions are met."""
    key, scene_tree, parent_node, parent_idx, priors = carry
    key, activate_key, props_key, color_key = jax.random.split(key, 4)
    shape_key, scale_key, place_key = jax.random.split(props_key, 3)

    activation_prob = priors.child_activation_prob_base * (
        priors.child_activation_prob_decay**child_num
    )
    activate_child = tfd.Bernoulli(probs=activation_prob).sample(
        seed=activate_key
    )

    def add_child_to_tree(tree):
        child_idx = jp.sum(tree.is_active)
        child_shape = sample_child_shape(shape_key, priors)
        child_scale, p_scale_vec = sample_child_scale(
            scale_key, priors, parent_node.relative_transform
        )
        translation, rotation = calculate_child_placement(
            place_key, priors, child_scale, p_scale_vec
        )
        child_color = determine_child_color(
            color_key, parent_node.material.color
        )
        child_transform = (
            SE3.translation(translation)
            @ rotation
            @ SE3.scaling(jp.full(3, child_scale))
        )
        updated_material = tree.material._replace(
            color=tree.material.color.at[child_idx].set(child_color)
        )
        updates = {
            "is_active": tree.is_active.at[child_idx].set(True),
            "parent_index": tree.parent_index.at[child_idx].set(parent_idx),
            "shape_type": tree.shape_type.at[child_idx].set(child_shape),
            "relative_transform": tree.relative_transform.at[child_idx].set(
                child_transform
            ),
            "material": updated_material,
        }
        return tree._replace(**updates)

    has_space = jp.sum(scene_tree.is_active) < MAX_NODES
    tree = jax.lax.cond(
        activate_child & has_space, add_child_to_tree, lambda t: t, scene_tree
    )
    return (key, tree, parent_node, parent_idx, priors), None


def gen_children_for_parent(key, scene_tree, parent_idx, priors):
    """Generates children for a single parent node."""
    parent_node = jax.tree.map(lambda x: x[parent_idx], scene_tree)

    def generate_children_loop(key, tree):
        final_carry, _ = jax.lax.scan(
            generate_child,
            (key, tree, parent_node, parent_idx, priors),
            jp.arange(MAX_CHILDREN),
        )
        updated_key, updated_tree, _, _, _ = final_carry
        return updated_key, updated_tree

    key, scene_tree = jax.lax.cond(
        parent_node.is_active,
        generate_children_loop,
        lambda key, tree: (key, tree),
        key,
        scene_tree,
    )
    return key, scene_tree


def generation_step(carry, parent_idx):
    """A single step in the scene generation process."""
    key, scene_tree, priors = carry
    key, scene_tree = gen_children_for_parent(
        key, scene_tree, parent_idx, priors
    )
    return (key, scene_tree, priors), None


@jax.jit
def sample_from_prior(key, priors):
    """Samples a scene tree from the defined prior distributions."""
    key, root_key = jax.random.split(key)
    scene_tree = create_empty_tree()
    scene_tree = init_root_node(root_key, scene_tree, priors)

    (final_key, final_tree, _), _ = jax.lax.scan(
        generation_step, (key, scene_tree, priors), jp.arange(MAX_NODES)
    )
    return final_tree


def calc_world_transforms(scene_tree):
    """Calculates the world transforms for each node in the scene tree."""
    world_transforms = scene_tree.relative_transform

    def update_transform(index, transforms):
        node = jax.tree.map(lambda x: x[index], scene_tree)
        parent_transform = transforms[node.parent_index]
        current_world_transform = parent_transform @ node.relative_transform
        return jax.lax.cond(
            node.parent_index >= 0,
            lambda t: t.at[index].set(current_world_transform),
            lambda t: t,
            transforms,
        )

    return jax.lax.fori_loop(1, MAX_NODES, update_transform, world_transforms)


@jax.jit
def compile_tree(scene_tree):
    """Compiles the scene tree into a format suitable for rendering."""
    world_transforms = calc_world_transforms(scene_tree)
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
    active_node_mask = scene_tree.is_active
    return scene_shapes, active_node_mask


def visualize_tree(scene_tree, filename="scene_tree"):
    """Generates a DOT graph visualization and saves it as a PDF."""
    dot = graphviz.Digraph(comment="Scene Tree")
    dot.attr("node", shape="box", style="rounded")

    for i in range(MAX_NODES):
        if scene_tree.is_active[i]:
            shape_type = int(scene_tree.shape_type[i])
            label = f"Node {i}\n({SHAPE_NAMES.get(shape_type, 'UNKNOWN')})"
            dot.node(str(i), label)
            parent_idx = scene_tree.parent_index[i]
            if parent_idx >= 0:
                dot.edge(str(parent_idx), str(i))

    output_filename = dot.render(
        filename, view=False, cleanup=True, format="pdf"
    )
    print(f"Generated tree visualization: {output_filename}")


def define_priors():
    """Defines the meta-priors for generating objects and rooms."""
    object_priors = MetaPriors(
        root_shape_probabilities=jp.array([0.7, 0.3, 0.0]),  # CUBE or SPHERE
        root_scale=1.0,
        child_scale_loc=0.6,
        child_scale_scale=0.2,
        child_scale_low=0.2,
        child_scale_high=0.9,
        camera_eye=jp.array([0.0, 0.0, -10.0]),
        light_position=jp.array([-10.0, 10.0, -10.0]),
        child_rotation_concentration=8.0,
        child_shape_probabilities=jp.array([0.6, 0.4, 0.0]),
        child_activation_prob_base=0.95,
        child_activation_prob_decay=0.6,
        face_probabilities=jp.full(6, 1.0 / 6.0),
    )
    room_priors = MetaPriors(
        root_shape_probabilities=jp.array([1.0, 0.0, 0.0]),  # Always a CUBE
        root_scale=3.0,
        child_scale_loc=0.25,
        child_scale_scale=0.1,
        child_scale_low=0.1,
        child_scale_high=0.4,
        camera_eye=jp.array([0.0, 0.0, -2.9]),
        light_position=jp.array([0.0, 2.5, 0.0]),
        child_rotation_concentration=20.0,
        child_shape_probabilities=jp.array([0.5, 0.5, 0.0]),
        child_activation_prob_base=0.9,
        child_activation_prob_decay=0.8,
        face_probabilities=jp.array(
            [0.01, 0.95, 0.01, 0.01, 0.01, 0.01]
        ),  # High prob for bottom face
    )
    return object_priors, room_priors


def gen_and_render_scenes(key, object_priors, room_priors, base_rays, H, W):
    """Generates and renders a set of scenes."""
    figure, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    figure.suptitle("Samples from Prior Predictive Distribution", fontsize=16)

    for i in range(6):
        print(f"Generating scene {i+1}/6...")
        key, sample_key, choice_key = jax.random.split(key, 3)

        choice = jax.random.randint(choice_key, (), 0, 2)
        priors = jax.lax.cond(
            choice == 0, lambda: object_priors, lambda: room_priors
        )

        scene_type = "object" if choice == 0 else "room"
        lights = [PointLight(jp.array([1.0, 1.0, 1.0]), priors.light_position)]
        camera_pose = SE3.view_transform(
            priors.camera_eye,
            jp.array([0.0, 0.0, 0.0]),
            jp.array([0.0, 1.0, 0.0]),
        )
        rays = transform_rays(camera_pose, base_rays[0], base_rays[1])

        render_fn = Render((H, W), camera_pose, rays, shadows=False)
        fast_render = jax.jit(render_fn)

        scene_tree = sample_from_prior(sample_key, priors)
        visualize_tree(scene_tree, filename=f"sample_tree_{i+1}")

        scene_shapes, mask = compile_tree(scene_tree)
        image, depth = fast_render(scene_shapes, mask, lights)

        axis1 = axes[i * 2]
        axis2 = axes[i * 2 + 1]

        axis1.imshow(jp.clip(image, 0, 1))
        axis1.set_title(f"Sample {i+1}: {scene_type} (RGB)")
        axis1.axis("off")

        axis2.imshow(depth, cmap="viridis")
        axis2.set_title(f"Sample {i+1}: Depth Map")
        axis2.axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def main():
    """Main function to run the scene generation and rendering."""
    H, W = 240, 320
    key = PRNGKey(777)

    object_priors, room_priors = define_priors()

    base_camera_pose = jp.eye(4)
    base_rays = build_rays((H, W), jp.pi / 4, base_camera_pose)

    gen_and_render_scenes(key, object_priors, room_priors, base_rays, H, W)


if __name__ == "__main__":
    main()
