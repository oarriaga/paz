import jax.numpy as jp

from paz.graphics.geometry import (
    compute_reflections_dot_eye,
    compute_hits_to_light,
)
import paz

from .patterns import compute_mesh_base_colors


def compute_mesh_colors(mesh, lights, points, normals, eyes, barycentric_u, barycentric_v):  # fmt: skip
    uv_barycentric = (barycentric_u, barycentric_v)
    base_colors = compute_mesh_base_colors(mesh, points, *uv_barycentric)
    colors = []
    material = mesh.material
    for light in lights:
        ambient = compute_ambient(material.ambient, base_colors, light, points)
        diffuse = compute_diffuse(material.diffuse, base_colors, light, points, normals)  # fmt: skip
        specular = compute_specular(material.specular, material.shininess, eyes, light, points, normals)  # fmt: skip
        colors.append(ambient + diffuse + specular)
    colors = jp.sum(jp.array(colors), axis=0)
    return colors


def compute_ambient(ambient, color, light, points):
    base_color = compute_base_color(color, light.intensity, points)
    return base_color * ambient


def compute_diffuse(diffuse, color, light, points, normals):
    hits_to_light = compute_hits_to_light(light.position, points)
    lambertian = paz.algebra.dot(hits_to_light, normals)
    lambertian = jp.maximum(lambertian, 0.0)
    lambertian = jp.expand_dims(lambertian, -1)
    base_color = compute_base_color(color, light.intensity, points)
    return base_color * diffuse * lambertian


def compute_specular(specular, shininess, eyes, light, points, normals):
    reflections = compute_reflections_dot_eye(light, points, normals, eyes)
    factor = jp.expand_dims(jp.power(reflections, shininess), -1)
    specular_color = light.intensity * specular * factor
    return specular_color


def compute_base_color(color, intensity, points):
    return color * intensity


def vertex_colors_to_face_colors(faces, vertex_colors):
    face_colors = jp.mean(vertex_colors[faces], axis=1)
    face_colors = jp.expand_dims(face_colors, axis=1)
    return face_colors
