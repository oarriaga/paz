# Differentiable rendering

Scenes built from `paz.graphics` shapes, materials and patterns, shown in
the interactive viewer or written to image files.

| Script | What it does |
| --- | --- |
| `axes.py` | Builds coordinate axes, saves the scene to `assets/axes` and opens the viewer. |
| `camera.py` | Builds a camera model from primitives and saves it to `assets/camera.json`. |
| `cornell_box_glass.py` | Renders a Cornell box and writes `cornell_box_mirror.png`. |
| `earth.py` | Maps `earthmap1k.jpg` onto a sphere and opens the viewer. |
| `globe.py` | Textured globe scene; saves it to `assets/globe`, reloads it and plots both renders. |
| `prototypes.py` | Renders one image per shape (`prototype_<index>.png`) and a mosaic `prototypes.png`. |
| `turtle.py` | Builds a turtle from primitives with a patterned shell and opens the viewer. |

## Run

```bash
python3 examples/differentiable_rendering/turtle.py
```

This opens the `paz.graphics.viewer` window with the turtle scene. The
scripts take no command line options.

## Requirements

- A display for the viewer and matplotlib windows.
- `earth.py` and `globe.py` load texture images from the working directory
  (`earthmap1k.jpg`, `gold_uv.png`, `wood_base.png`, `wood_table.png`,
  `wall_uv.jpg`). These files are not shipped in this folder.
