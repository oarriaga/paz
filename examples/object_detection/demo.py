import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"
import paz

camera = 0
pipeline = paz.applications.SSD300VOC()
# pipeline = paz.applications.SSD512COCO()
camera = paz.Camera(identifier=camera)
player = paz.VideoPlayer((480, 640), pipeline, camera)
player.run()
