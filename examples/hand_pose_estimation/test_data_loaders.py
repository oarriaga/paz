from data_loaders import HandPoseLoader
from backend import to_homogeneous_coordinates, normalize_keypoints

data_loader = HandPoseLoader(
    '/home/dfki.uni-bremen.de/jbandlamudi/DFKI_Work/RHD_published_v2/')


def test_image_loading(image_path):
    image = data_loader.load_images(image_path)
    assert image.shape == data_loader.image_size


def test_segmentation_map_loading(segmentation_path):
    segmentation_mask = data_loader.load_images(segmentation_path)
    assert segmentation_mask.shape == data_loader.image_size


def test_conversion_to_homogeneous_coordinates(vector):
    homogeneous_vector = to_homogeneous_coordinates(vector)
    assert len(homogeneous_vector) == 4


def test_keypoint_normalization(keypoints):
    keypoint_scale, norm_keypoints = normalize_keypoints(keypoints)
