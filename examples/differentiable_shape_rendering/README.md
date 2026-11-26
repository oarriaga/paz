# Differentiable shape rendering

Scenes made of `paz.graphics` primitives (spheres, planes, cubes, cones,
cylinders) with Phong and Cook-Torrance materials, lights and shadows.

| Script | What it does |
| --- | --- |
| `axes.py` | Coordinate axes; saves the scene to `assets/axes` and shows the render. |
| `balloons.py` | Balloons built from spheres, knots and strings; writes a PNG. |
| `build_true_image.py` | Floor, sphere and cone with procedural textures; writes `true_image.png`. |
| `camera.py` | Camera model built from primitives; saves it to `assets/camera`. |
| `cook_torrance_example.py` | Dielectric and metal spheres; writes `cook_torrance.png`. |
| `cornell_box.py` | Cornell box; writes `cornell_box_mirror.png` and opens the viewer. |
| `mask_boxes.py` | Renders random subsets of cubes using the shape mask and plots each. |
| `optimize_scene.py` | Optimizes material parameters against a target image; writes to a timestamped folder under `experiments`. |
| `phong_realism_levels.py` | Same scene from ambient-only up to area lights, shadows and bounces. |
| `phong_reflection_model.py` | Balloon scene for the Phong reflection model; writes a PNG. |
| `phong_vs_cook_torrance.py` | Writes `phong.png`, `cook_torrance.png` and `phong_vs_cook_torrance.png`. |
| `primitives.py` | Shows renders of the primitives with a checkered pattern. |
| `ray_tracing.py` | Diagram scene of camera, image plane, rays and shadow rays; writes `ray_tracing.png`. |
| `realistic_primitives.py` | Sphere, cylinder, cone and cube resting on a floor; writes images. |
| `shininess_vs_specular.py` | Grid over shininess and specular values; writes `mosaic.png`. |
| `sphere_and_plane.py` | Minimal sphere on a plane with one point light in the viewer. |
| `turtle.py` | Turtle built from primitives with a patterned shell; shows the render. |
| `visualize_scene_gradients.py` | Compares autodiff and finite-difference image gradients; writes `color_gradients_scene.png`. |

## Run

```bash
python3 examples/differentiable_shape_rendering/sphere_and_plane.py
```

This opens the `paz.graphics.viewer` window at 240x320 with the sphere and
plane. Scripts that call `paz.image.write` save into the working directory.

## Requirements

A display for the scripts that open the viewer, `paz.image.show` or
matplotlib windows.
