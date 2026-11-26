# Structure from motion

Reconstructs a sparse colored 3D pointcloud from a sequence of images.

| Script | What it does |
| --- | --- |
| `demo.py` | Loads the images in a folder, reconstructs the scene and plots the 3D points. |
| `demo_live.py` | Captures frames from a camera, saves every `--frame_skip`-th frame as PNG and reconstructs from them. |
| `pipeline.py` | The reconstruction pipeline (`reconstruct_scene`). |
| `backend.py` | OpenCV based helpers used by the pipeline. |
| `geometry.py` | JAX geometry functions used by the pipeline. |
| `geometry_test.py` | Tests for `geometry.py`. |

## Run

```bash
cd examples/structure_from_motion
python3 demo.py --images_path datasets/images --HFOV 70 --match_ratio 0.75
```

Other options: `--residual_thresh`, `--correspondence_thresh`, `--seed`.
`demo_live.py` takes `--camera_id` and `--frame_skip` instead of
`--images_path`.

The result is a matplotlib 3D plot of the reconstructed points with their
colors.

## Requirements

- A folder of images of one scene at `--images_path`; none is shipped.
- A display, and a camera for `demo_live.py`.
