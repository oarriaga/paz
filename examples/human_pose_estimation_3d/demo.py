from pipeline import SIMPLEBASELINES
from paz.applications import HigherHRNetHumanPose2D
from paz.backend.image import load_image
from linear_model import SIMPLE_BASELINE
from backend import solve_translation, visualize

path = 'test_image.jpg'
image = load_image(path)
image_height, image_width = image.shape[:2]
keypoints2D = HigherHRNetHumanPose2D()
model = SIMPLE_BASELINE(16, 3, 1024, (32,), 2, 1)
model.load_weights('weights.h5')
pipeline = SIMPLEBASELINES(keypoints2D, model)
keypoints = pipeline(image)
keypoints2D, joints3D, keypoints3D, poses3D = solve_translation(
    keypoints['keypoints2D'], keypoints['keypoints3D'], image_height,
    image_width)
visualize(keypoints2D, joints3D, keypoints3D, poses3D)
