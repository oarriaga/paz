# Plane detection

`demo.py` converts a FewSOL depth image into a pointcloud, fits the
supporting plane, plots the points with the plane and its normal, and
renders a `paz.graphics.Plane` at the fitted pose.

## Run

```bash
python3 examples/plane_detection/demo.py
```

The script has no command line options; the shot, class, image size
(480x640) and maximum depth are set as variables in the file.

## Output

- Matplotlib 3D figures of the pointcloud, the plane and its normal.
- A window with a 1x2 mosaic of the dataset image and the rendered plane.

## Requirements

- The FewSOL dataset, which is not shipped with the repository.
- A display for the matplotlib and image windows.
