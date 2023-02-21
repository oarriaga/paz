from paz.abstract import Processor
import paz.processors as pr
from keypoints_processors import SimpleBaselines3D
from human36m import data_mean2D, data_stdev2D, data_mean3D, data_stdev3D, \
	dim_to_use3D


class SIMPLEBASELINES(Processor):
	def __init__(self, estimate_keypoints_2D, estimate_keypoints_3D):
		self.estimate_keypoints_2D = estimate_keypoints_2D
		self.estimate_keypoints_3D = estimate_keypoints_3D
		self.wrap = pr.WrapOutput(['keypoints2D', 'keypoints3D'])

	def call(self, image):
		inferences2D = self.estimate_keypoints_2D(image)
		keypoints2D = inferences2D['keypoints']
		baseline_model = SimpleBaselines3D(self.estimate_keypoints_3D,
		                                   data_mean2D, data_stdev2D,
		                                   data_mean3D, data_stdev3D,
		                                   dim_to_use3D)
		keypoints2D, keypoints3D = baseline_model(keypoints2D)
		return self.wrap(keypoints2D, keypoints3D)
