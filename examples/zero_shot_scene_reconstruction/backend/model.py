from functools import partial

import jax
import jax.numpy as jp
from jax.image import resize
import paz

from paz.graphics import render_masks
from .mesh import append_mesh
from .scene import to_render_material


def preprocess_observations(true_image, true_depth, shot_masks, image_shape):
    true_image = resize(true_image, (*image_shape, 3), "bilinear")
    true_depth = resize(true_depth, (*image_shape, 1), "nearest")
    true_masks = []
    for true_mask in shot_masks.copy():
        true_mask = 255.0 * true_mask.astype(float)
        true_mask = resize(true_mask, (*image_shape, 1), "bilinear")
        true_mask = true_mask.astype(bool).astype(float)
        true_masks.append(true_mask)
    return true_image, true_depth, jp.array(true_masks)


def split_batched_tree(tree):
    get_item = lambda i: jax.tree.map(lambda leaf: leaf[i], tree)
    return [get_item(i) for i in range(tree.color.shape[0])]


def parameters_to_scene(geometry, floor_material, materials, shapes):
    initial_scale, directions3D, origins3D = geometry
    scale_vectors, distances = shapes
    move = lambda direction, origin, distance: origin + distance * direction
    points3D = jax.vmap(move)(directions3D, origins3D, distances)
    shifts = jax.vmap(paz.SE3.translation)(points3D)
    scales = jax.vmap(paz.SE3.scaling)(scale_vectors)
    final_scale = jax.vmap(jp.matmul)(scales, initial_scale)
    transforms = jax.vmap(jp.matmul)(shifts, final_scale)
    floor_material = to_render_material(floor_material)
    materials = split_batched_tree(materials)
    materials = [to_render_material(material) for material in materials]
    spheres = [paz.graphics.Sphere(t, m) for t, m in zip(transforms, materials)]
    floor = paz.graphics.Plane(material=floor_material)
    scene = paz.graphics.Scene([*spheres, floor])
    scales3D = jax.vmap(jp.diag)(final_scale)[:, :3]
    return scene, transforms, points3D, scales3D


def build_shape_model(camera, geometry, shadows):
    image_shape, y_FOV, world_to_camera, min_depth, max_depth = camera

    def model(lights, floor_material, materials, scale_vectors, distances):
        shapes = (scale_vectors, distances)
        scene_args = (geometry, floor_material, materials, shapes)
        result = parameters_to_scene(*scene_args)
        scene, transforms, points3D, final_scale_vectors = result
        view_args = image_shape, y_FOV, world_to_camera, scene, None, lights
        view_args = view_args + ((1, 1), image_shape[0] * image_shape[1])
        image, depth = paz.graphics.render(*view_args, shadows)
        num_objects = len(scene.nodes) - 1
        mask_args = image_shape, y_FOV, world_to_camera, scene, lights
        depth_range = min_depth, max_depth
        masks = render_masks(
            *mask_args,
            depth_range,
            (1, 1),
            image_shape[0] * image_shape[1],
            num_objects=num_objects,
            shadows=shadows,
        )
        depth = jp.expand_dims(depth, axis=-1)
        aux = {
            "transforms": transforms,
            "translations": points3D,
            "final_scale_vectors": final_scale_vectors,
        }
        return image, depth, masks, aux

    return model


def build_mesh_model(camera, meshes, mesh_weights, floor, lights):
    image_shape, y_FOV, world_to_camera = camera[0], camera[1], camera[2]
    min_depth, max_depth = camera[3], camera[4]
    tile_shape, chunk_size = camera[5], camera[6]
    num_objects = meshes.vertices.shape[0]
    num_v = meshes.vertices.shape[1]
    num_f = meshes.faces.shape[1]
    num_e = meshes.edges.shape[1]
    filled_floor = paz.graphics.mesh.fill_mesh(floor, num_v, num_f, num_e)
    tile = tuple(tile_shape)

    def model(cage_vertices):
        deform = partial(paz.cage.control_mesh, mesh_weights)
        deformed_verts = jax.vmap(deform)(cage_vertices)
        updated = meshes._replace(vertices=deformed_verts)
        scene_meshes = append_mesh(updated, filled_floor)
        all_mask = jp.ones(num_objects + 1, dtype=bool)
        render_args = image_shape, y_FOV, world_to_camera, scene_meshes
        render_args = render_args + (all_mask, lights, tile, chunk_size)
        image, depth = paz.graphics.mesh.render(*render_args)
        depth_range = (min_depth, max_depth)
        mask_args = image_shape, y_FOV, world_to_camera, updated, lights
        mask_args = mask_args + (depth_range, tile, chunk_size)
        masks = paz.graphics.mesh.render_masks(*mask_args)
        depth = jp.expand_dims(depth, axis=-1)
        aux = {"meshes": updated, "vertices": deformed_verts}
        aux["faces"] = meshes.faces
        return image, depth, masks, aux

    return model
