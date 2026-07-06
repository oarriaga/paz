# PIX2POSE — 6D object pose (power drill)

Synthetic-data PIX2POSE, generated with PAZ's internal differentiable mesh
renderer (`paz.graphics.mesh`) instead of pyrender. The model regresses a dense
object-coordinate (NOCS) map from an RGB crop; 6D pose is recovered from the map
with PnP-RANSAC.

All scripts default to `KERAS_BACKEND=jax`.

## How the data is generated

For each random camera pose the renderer produces two aligned outputs:

- **RGB input** — the textured drill (`mesh_renderer`), then domain-randomized
  (`paz.image.randomize_rendered_image`: VOC background, occlusions, blur,
  color jitter). This is what the network sees.
- **NOCS label** — per-pixel object-frame 3D coordinates, computed analytically
  from the ray tracer's barycentric hits (`paz.graphics.mesh.render_coordinates`
  → normalize by the mesh bounding box). Channels are `[X, Y, Z]` in `[0, 1]`,
  confined to the object silhouette. A 4th alpha channel (the mask) is appended
  so `WeightedReconstruction` can up-weight object pixels.

The drill mesh (`035_power_drill`) ships with ~524k faces, which renders at
~0.8 s/sample on GPU. It is decimated to `--target_faces` (default 20000) with
trimesh quadric decimation; the baked texture colors are transferred to the
decimated vertices by nearest neighbor. This drops rendering to ~32 ms/sample
(~25× faster) with no visible quality loss and the object extents preserved to
< 0.1 mm (so pose scaling is unaffected). Data is still rendered once and
cached; augmentation runs per batch.

## Scripts

```bash
python visualize_pipeline.py          # contact sheet: input | NOCS | mask (+timing)
python train.py                       # render, cache, train UNET_VGG16
python demo.py                        # NOCS -> PnP-RANSAC -> projected 3D box
python demo.py --weights <path.h5>    # use trained weights instead of GT NOCS
python validate.py --weights <path.h5>  # NOCS MAE + pose error on held-out poses
```

On the decimated (20k) drill, ~26 epochs reach a NOCS foreground MAE of ~0.055
and a median pose reprojection error of ~2 px over held-out poses.

`visualize_pipeline.py` and `demo.py` validate the pipeline without training:
they check the NOCS label is 0 outside the mask and spans `[0, 1]` inside, and
that recovered poses reproject with sub-pixel error.

## Library pieces used

- `paz.graphics.mesh.render_coordinates` — object-coordinate (NOCS) render.
- `paz.losses.WeightedReconstruction` — masked L1 on the `[RGB, alpha]` label.
- `paz.applications.pose_estimators.solve_PnP_RANSAC` — pose from correspondences.
- `paz.models.UNET_VGG16` — the coordinate-map regressor.
