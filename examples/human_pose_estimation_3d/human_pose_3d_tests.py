import pytest
import numpy as np
from paz.backend.image import load_image
from paz.applications import HigherHRNetHumanPose2D
from human36m import filter_keypoints2D, unnormalize_data
from linear_model import SIMPLE_BASELINE
from backend import standardize
from human36m import data_mean2D, data_stdev2D, data_mean3D, \
	data_stdev3D, dim_to_use3D


def get_poses2D(image):
	detect = HigherHRNetHumanPose2D()
	poses2D = detect(image)
	poses2D = poses2D['keypoints']
	poses2D = filter_keypoints2D(poses2D)
	return poses2D


def get_poses3D(poses2D, model):
	norm_data = standardize(poses2D, data_mean2D, data_stdev2D)
	poses3D = model.predict(norm_data)
	poses3D = unnormalize_data(poses3D, data_mean3D, data_stdev3D,
	                           dim_to_use3D)
	return poses3D


@pytest.fixture
def model():
	model = SIMPLE_BASELINE(16, 3, 1024, (32,), 2, 1)
	model.load_weights('weights.h5')
	return model


@pytest.fixture
def image_with_multiple_persons_A():
	path = 'test_image.jpg'
	image = load_image(path)
	return image


@pytest.fixture
def image_with_single_person_B():
	path = 'test.jpg'
	image = load_image(path)
	return image


@pytest.fixture
def keypoints3D_multiple_persons():
	return np.array(
		[[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, -1.14726820e+02,
		1.21343876e+01, -6.79812245e+01, -1.35863824e+02, 4.19467606e+02,
		9.37677967e+01, -2.14377161e+02, 8.06703618e+02, 2.89143238e+02,
		-1.15560633e+01, 7.42149725e+02, 1.66477287e+02, -1.18447102e+01,
		7.36763064e+02, 1.65182437e+02, 1.14726152e+02, -1.21343424e+01,
		6.79808542e+01, 1.27938000e+02, 4.13484511e+02, 1.91325905e+02,
		9.55896841e+01, 8.01181935e+02, 3.89558022e+02, -1.12642400e+01,
		7.48636864e+02, 1.66665977e+02, -1.14090840e+01, 7.36435064e+02,
		1.63713810e+02, 1.21660909e-03, -8.60110488e-02, -1.93000547e-02,
		4.52335487e+00, -2.49469926e+02, -6.15344365e+01, 2.27819061e+01,
		-4.84445853e+02, -1.72157425e+02, 5.55437990e+01, -5.64491792e+02,
		-2.90167339e+02, 2.12989218e+01, -6.77382407e+02, -2.97420008e+02,
		5.67537804e+00, -4.35088906e+02, -9.76974016e+01, 1.81741537e+02,
		-4.95856935e+02, -1.24252892e+02, 2.18840236e+02, -2.77173386e+02,
		-1.83449219e+02, -3.68658086e+01, -2.67203020e+02, -2.90879981e+02,
		1.26585821e+00, -1.20170579e+02, -2.82526049e+01, 1.57900698e+00,
		-1.51780249e+02, -3.52080548e+01, 8.84543990e-01, -1.07795356e+02,
		-2.56307189e+01, 8.84543990e-01, -1.07795356e+02, -2.56307189e+01,
		5.67537804e+00, -4.35088906e+02, -9.76974016e+01, -1.35515689e+02,
		-4.50778918e+02, -2.07550560e+02, -1.54306546e+02, -1.97581166e+02,
		-2.56477770e+02, 6.74229966e+01, -2.08812250e+02, -3.18276339e+02,
		8.70569018e-01, -1.68664569e+02, -3.73902498e+01, 1.39982513e+00,
		-2.00884252e+02, -4.47207875e+01, 5.24591118e-01, -1.65867774e+02,
		-3.68342864e+01, 5.24591118e-01, -1.65867774e+02, -3.68342864e+01]
		])


@pytest.fixture
def keypoints2D_multiple_persons():
	return np.array(
		[297.97265625, 278.81640625, 270.41210938, 279.08398438, 265.59570312,
		384.50976562, 258.10351562, 480.30273438, 325.53320312, 278.54882812,
		330.34960938, 382.90429688, 334.63085938, 477.09179688, 297.97265625,
		212.45703125, 297.97265625, 146.09765625, 295.02929688, 86.96289062,
		340.51757812, 142.08398438, 348.54492188, 198.27539062, 283.79101562,
		195.06445312, 255.42773438, 150.11132812, 250.07617188, 211.65429688,
		308.40820312, 210.58398438])


@pytest.fixture
def keypoints3D_single_person():
	return np.array(
		[[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, -1.01613821e+02,
		-1.57045238e+00, -7.83579210e+01, -1.28771002e+02, 3.03492329e+02,
		-1.30643458e+02, -2.01166429e+02, 6.75673518e+02, -1.05658365e+02,
		-1.15560633e+01, 7.42149725e+02, 1.66477287e+02, -1.18447102e+01,
		7.36763064e+02, 1.65182437e+02, 1.01614079e+02, 1.57052265e+00,
		7.83575426e+01, 7.93183401e+01, 3.39041783e+02, 1.92607564e+01,
		2.15506220e+01, 6.78448656e+02, 4.92480396e+01, -1.12642400e+01,
		7.48636864e+02, 1.66665977e+02, -1.14090840e+01, 7.36435064e+02,
		1.63713810e+02, 1.21660909e-03, -8.60110488e-02, -1.93000547e-02,
		-6.79559464e+00, -2.54716530e+02, -5.36947436e+01, 7.82148523e+00,
		-4.86618964e+02, -1.56544606e+02, 2.98269877e+01, -6.25791310e+02,
		-2.51405775e+02, 2.25821517e+01, -7.53364099e+02, -2.59546802e+02,
		5.67537804e+00, -4.35088906e+02, -9.76974016e+01, 2.02423399e+02,
		-5.18830395e+02, -8.32461342e+01, 2.62380803e+02, -2.59514591e+02,
		2.98386633e+01, 1.97672719e+02, -4.68176736e+01, -3.21776639e+01,
		1.26585821e+00, -1.20170579e+02, -2.82526049e+01, 1.57900698e+00,
		-1.51780249e+02, -3.52080548e+01, 8.84543990e-01, -1.07795356e+02,
		-2.56307189e+01, 8.84543990e-01, -1.07795356e+02, -2.56307189e+01,
		5.67537804e+00, -4.35088906e+02, -9.76974016e+01, -1.93475903e+02,
		-5.00181221e+02, -2.05483820e+02, -2.76693873e+02, -2.39491927e+02,
		-1.01124915e+02, -2.05551736e+02, 7.92245260e+00, -4.89243838e+01,
		8.70569018e-01, -1.68664569e+02, -3.73902498e+01, 1.39982513e+00,
		-2.00884252e+02, -4.47207875e+01, 5.24591118e-01, -1.65867774e+02,
		-3.68342864e+01, 5.24591118e-01, -1.65867774e+02, -3.68342864e+01]
		 ])


@pytest.fixture
def keypoints2D_single_person():
	return np.array(
		[377.19726562, 627.07519531, 321.89941406, 625.61035156, 314.57519531,
		773.55957031, 282.34863281, 922.24121094, 432.49511719, 628.54003906,
		414.18457031, 780.15136719, 410.52246094, 919.31152344, 375.91552734,
		506.95800781, 374.63378906, 386.84082031, 382.69042969, 271.11816406,
		469.11621094, 387.57324219, 507.93457031, 515.01464844, 484.49707031,
		618.28613281, 280.15136719, 386.10839844, 239.13574219, 516.47949219,
		275.75683594, 625.61035156])


def test_simple_baselines_multiple_persons(image_with_multiple_persons_A,
                                           keypoints3D_multiple_persons,
                                           keypoints2D_multiple_persons,
                                           model):
	poses2D = get_poses2D(image_with_multiple_persons_A)
	assert np.allclose(poses2D[0], keypoints2D_multiple_persons)
	poses3D = get_poses3D(poses2D, model)
	assert np.allclose(poses3D[0], keypoints3D_multiple_persons)


def test_simple_baselines_single_persons(image_with_single_person_B,
                                         keypoints3D_single_person,
                                         keypoints2D_single_person, model):
	poses2D = get_poses2D(image_with_single_person_B)
	assert np.allclose(poses2D, keypoints2D_single_person)
	poses3D = get_poses3D(poses2D, model)
	assert np.allclose(poses3D, keypoints3D_single_person)
