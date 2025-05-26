import os

os.environ["KERAS_BACKEND"] = "jax"
import paz


pipeline = paz.applications.DetectMiniXceptionFER()
camera = paz.Camera(identifier=0)
player = paz.VideoPlayer((480, 640), pipeline, camera)
player.run()
