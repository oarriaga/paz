from collections import namedtuple

import jax
import jax.numpy as jp
import paz

from . import plane as plane_backend

MaterialParameters = namedtuple(
    "MaterialParameters",
    ["color", "ambient", "diffuse", "specular", "shininess"],
)

def build_camera_pose(pointcloud, intrinsics, image_size, max_depth, seed):
    H, W = image_size
    pointcloud = paz.pointcloud.bound(pointcloud, max_depth)
    plane_camera_openCV, inner_mask = plane_backend.fit_RANSAC(seed, pointcloud)
    pose_args = (H, W, intrinsics, plane_camera_openCV)
    world_to_camera_opencv = plane_backend.build_plane_pose(*pose_args)
    rot = paz.SE3.rotation_x(jp.pi / 2.0)
    world_to_camera_opencv = world_to_camera_opencv @ rot
    camera_openCV_to_plane = jp.linalg.inv(world_to_camera_opencv)
    openGL_to_openCV = paz.SE3.rotation_x(jp.pi)  # 180° around X flips Y, Z
    camera_to_plane = camera_openCV_to_plane @ openGL_to_openCV
    camera_origin = camera_to_plane[:3, -1]
    world_to_camera_opengl = jp.linalg.inv(camera_to_plane)
    poses = (world_to_camera_opengl, camera_origin, inner_mask)
    return (*poses, world_to_camera_opencv)



def initialize_light_intensity(key, min_value, max_value):
    return jax.random.uniform(key, (3,), minval=min_value, maxval=max_value)


def initialize_light_position(key):
    key_0, key_1 = jax.random.split(key)
    x, y = jax.random.uniform(key_0, (2,), minval=-2.0, maxval=2.0)
    z = jax.random.uniform(key_1, minval=0.15, maxval=1.0)
    return jp.array([x, z, y])


def initialize_light(key, min_intensity, max_intensity):
    key_p, key_i = jax.random.split(key)
    position = initialize_light_position(key_p)
    intensity = initialize_light_intensity(key_i, min_intensity, max_intensity)
    return paz.graphics.PointLight(intensity, position)


def initialize_lights(key, num_lights, min_intensity, max_intensity):
    lights = []
    for light_key in jax.random.split(key, num_lights):
        lights.append(initialize_light(light_key, min_intensity, max_intensity))
    return lights


def initialize_floor_material(ambient, diffuse, specular, shininess):
    args = tuple(jp.array(x) for x in [ambient, diffuse, specular, shininess])
    return MaterialParameters(jp.ones(3), *args)


def to_material(mask, material_args, image):
    color = paz.mask.to_rgb(image, mask)
    return MaterialParameters(color, *material_args)


def initialize_shape_materials(image, masks, shading):
    material_args = [jp.array(x) for x in shading]
    return jax.vmap(paz.lock(to_material, material_args, image))(masks)


def to_render_material(material):
    args = (material.color, material.ambient, material.diffuse)
    extra = (material.specular, material.shininess, jp.array(0.0))
    return paz.graphics.Material(*args, *extra, jp.array(0.0), jp.array(1.0))


def move_in_line(direction3D, origin3D, distance):
    return (distance * direction3D) + origin3D


def initialize_lines(camera_origin, points3D):
    origins3D = jp.repeat(camera_origin[None], len(points3D), axis=0)
    directions3D = points3D - origins3D
    distances = jp.expand_dims(jp.linalg.norm(directions3D, axis=1), axis=1)
    directions3D = directions3D / distances
    return directions3D, origins3D, distances


def move_in_lines(directions3D, origins3D, distances):
    move = jax.vmap(move_in_line)
    return move(directions3D, origins3D, distances)
