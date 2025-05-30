import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"

from pipeline import DetectFaceKeypointNet2D32
import paz

camera = 0
pipeline = DetectFaceKeypointNet2D32()
camera = paz.Camera(identifier=camera)
player = paz.VideoPlayer((480, 640), pipeline, camera)
player.run()


# from pipeline import FaceKeypointNet2D32
# import numpy as np

# model = FaceKeypointNet2D32(draw=None)
# image = np.zeros((100, 100, 3)).astype(np.float32)
# keypoints, image = model(image)
# print(keypoints.shape)
