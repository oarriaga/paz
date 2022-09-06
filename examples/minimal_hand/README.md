### This example detects hand pose from an image.

To test the live hand pose estimation from camera, run:
```py
python demo.py
```

To test the hand pose estimation on image, run:
```py
python demo_image.py
```

To test the live hand closure status with the pose estimation from camera, run:
```py
python is_open_demo.py
```

To test the live hand pose estimation from camera and visualize keypoints in 3D, run (This module has an extra dependency of matplotlib): 
```py
python demo3D.py
```

### Additional notes
To test a more robust hand pose estimation and open / close classification try out the "paz/examples/hand_detection/pose_demo.py"

