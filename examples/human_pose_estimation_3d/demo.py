from pipeline import SIMPLEBASELINES
from paz.applications import HigherHRNetHumanPose2D
from paz.backend.image import load_image
from linear_model import SIMPLE_BASELINE
from helper_functions import solve_translation


path = 'test_image.jpg'
image = load_image(path)
image_h, image_w = image.shape[:2]
keypoints2D = HigherHRNetHumanPose2D()
model = SIMPLE_BASELINE(1024, (32,), 2, True, True, True, 1)
model.load_weights('weights.h5')
pipeline = SIMPLEBASELINES(keypoints2D, model)
keypoints = pipeline(image)
solve_translation(keypoints['keypoints2D'], keypoints['keypoints3D'], image_h,
                  image_w)
