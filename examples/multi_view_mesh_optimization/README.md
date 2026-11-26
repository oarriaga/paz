# Multi-view mesh optimization

`optimize_mesh.py` deforms a sphere mesh until its renders match color
images and masks of a textured cow mesh rendered from several camera
poses. The loss combines RGB, silhouette, edge, normal and Laplacian terms.

## Run

```bash
python3 examples/multi_view_mesh_optimization/optimize_mesh.py \
    --max_steps 2000 --num_views 20 --image_size 128
```

Other options: `--mesh_dir`, `--output_dir`, `--seed`, `--y_fov`,
`--mask_sigma`, `--mask_bin_size`, `--max_faces_per_bin`, `--rgb_weight`,
`--silhouette_weight`, `--edge_weight`, `--normal_weight`,
`--laplacian_weight`, `--save_every`, `--video_fps`.

## Output

Written to `--output_dir` (default `results` in this folder):
`comparison_view.png`, `comparison_mask.png`, `trace_grid.png`,
`trace_mask_grid.png`, `step_images/step_<step>.png`, `optimization.mp4`,
`losses.csv`, `losses.png`, `metrics.json` and `final_model.obj`.

## Requirements

- `cow.obj` and `cow_texture.png` inside `--mesh_dir` (default
  `data/cow_mesh` in this folder). These files are not shipped.
- `trimesh` to load the mesh.
