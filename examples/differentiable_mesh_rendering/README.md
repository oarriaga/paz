# Differentiable mesh rendering

Rendering triangle meshes with `paz.graphics`, including tiled rendering
and a gradient through vertex positions.

| Script | What it does |
| --- | --- |
| `render_bunny.py` | Loads `bunny.obj`, places it on a plane with one point light and opens a viewer rendered in 4x4 tiles. |
| `render_large_mesh.py` | Renders a mesh in one pass and in tiles, then computes a gradient with respect to the vertices. |

## Run

```bash
python3 examples/differentiable_mesh_rendering/render_large_mesh.py
```

The script writes `render_full.png` and `render_tiled.png` to the working
directory and prints progress. Neither script takes command line options.

## Requirements

- `render_bunny.py` reads `bunny.obj` from this folder. The file is not
  shipped; place a mesh with that name here first.
- A display for the viewer in `render_bunny.py`.
