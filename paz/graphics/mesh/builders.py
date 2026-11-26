import jax.numpy as jp


def build_cube(size=1.0):
    import trimesh
    import numpy as onp

    mesh = trimesh.creation.box(extents=[size, size, size])
    vertices = jp.array(mesh.vertices.view(onp.ndarray))
    faces = jp.array(mesh.faces.view(onp.ndarray))
    edges = jp.array(mesh.edges.view(onp.ndarray))
    return vertices, faces, edges


def build_sphere(radius=1.0, subdivisions=3):
    import trimesh
    import numpy as onp

    mesh = trimesh.creation.icosphere(subdivisions, radius)
    vertices = jp.array(mesh.vertices.view(onp.ndarray))
    faces = jp.array(mesh.faces.view(onp.ndarray))
    edges = jp.array(mesh.edges.view(onp.ndarray))
    return vertices, faces, edges


def load_mesh(filepath):
    import trimesh
    import numpy as onp

    mesh = trimesh.load(filepath)
    vertices = jp.array(mesh.vertices.view(onp.ndarray))
    faces = jp.array(mesh.faces.view(onp.ndarray))
    vertex_colors = to_color_visual(mesh.visual).vertex_colors[:, :3]
    vertex_colors = jp.array(vertex_colors.view(onp.ndarray))
    vertex_colors = vertex_colors / 255.0
    return vertices, faces, vertex_colors


def to_color_visual(visual):
    if visual.kind == "texture":
        color_visual = visual.to_color()
    else:
        color_visual = visual
    return color_visual
