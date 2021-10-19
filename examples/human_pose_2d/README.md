### Human pose estimation 2D

#### Directory structure for test images and the output:

```
human_pose_2d
|
|__images
|
|__output
   |
   |__result.jpg
```

* Provide the name of the image in the `image_path` of `demo.py`.
```
image_path = os.path.join(args.image_path, <image name>)
```

#### To run demo for a test image:
```
python demo.py
```

#### To run demo for a live video:
```
python demo_video.py.py
```
