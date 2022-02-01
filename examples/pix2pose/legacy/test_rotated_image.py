import numpy as np
import os
import glob
from paz.backend.image import show_image, resize_image
from paz.models import UNET_VGG16
from paz.abstract import GeneratingSequence

from paz.backend.camera import Camera
from pipelines import Pix2Pose
from pipelines import EstimatePoseMasks
from pipelines import DomainRandomization
from backend import build_rotation_matrix_z, rotate_image
from scenes import PixelMaskRenderer
from backend import build_rotation_matrix_x, build_rotation_matrix_y
from backend import denormalize_points2D
from processors import SolveChangingObjectPnPRANSAC
from paz.backend.quaternion import rotation_vector_to_quaternion
from paz.abstract.messages import Pose6D
from backend import build_cube_points3D
from backend import draw_poses6D
from paz.backend.image import load_image
from backend import draw_masks


scale = 4
H, W, num_channels = image_shape = [128, 128, 3]
root_path = os.path.expanduser('~')
background_wildcard = '.keras/paz/datasets/voc-backgrounds/*.png'
background_wildcard = os.path.join(root_path, background_wildcard)
image_paths = glob.glob(background_wildcard)

path_OBJ = 'single_solar_panel_02.obj'
path_OBJ = os.path.join(root_path, path_OBJ)
num_occlusions = 1
viewport_size = image_shape[:2]
y_fov = 3.14159 / 4.0
distance = [1.0, 1.0]
light = [1.0, 30]
top_only = False
roll = 3.14159
shift = 0 # %0.05
batch_size = 32
steps_per_epoch = 1000

image_size = [128, 128]
focal_length = image_size[1]
image_center = (image_size[1] / 2.0, image_size[0] / 2.0)
camera_intrinsics = np.array([[focal_length, 0, image_center[0]],
                              [0, focal_length, image_center[1]],
                              [0, 0, 1]])


image_shape = (128, 128, 3)
num_classes = 3
model = UNET_VGG16(num_classes, image_shape, freeze_backbone=True)
model.load_weights('weights/UNET-VGG_solar_panel_canonical_13.hdf5')
object_sizes_list = [15000, 15000, 2000]
object_sizes = np.array(object_sizes_list)
cube_points = object_sizes
cube_points3D = build_cube_points3D(*object_sizes)
epsilon = 0.15
estimate_keypoints = Pix2Pose(model, object_sizes, epsilon, True)
print(object_sizes)
predict_pose = SolveChangingObjectPnPRANSAC(camera_intrinsics, 5, 100)


def quick_pose(image):
    image = resize_image(image, (128, 128))
    # show_image(resize_image(image, (256 * 3, 256 * 3)))
    keypoints = estimate_keypoints(image)
    points2D = keypoints['points2D']
    points3D = keypoints['points3D']
    # points3D[:, 2:3] = 0.0
    points2D = denormalize_points2D(points2D, 128, 128)
    success, rotation, translation = predict_pose(points3D, points2D)
    quaternion = rotation_vector_to_quaternion(rotation)
    pose6D = Pose6D(quaternion, translation, 'solar_panel')
    poses6D = [pose6D]
    # show_image(image)
    points = [[points2D, points3D]]
    image = draw_masks(image, points, object_sizes)
    image = image.astype('float')
    image = draw_poses6D(image, poses6D, cube_points3D, camera_intrinsics)
    image = image.astype('uint8')
    image = resize_image(image, (256 * 3, 256 * 3))
    show_image(image)


image = load_image('images/zed_left_1011.png')
image = image[250:800, 250:850, :]
H, W, num_channels = image.shape
# image = resize_image(image, (W * 20, H * 20))
quick_pose(image)

image = load_image('images/MicrosoftTeams-image.png')
quick_pose(image)

image = load_image('images/zed_left_705.png')
image = image[250:1080, 250:1400, :]
quick_pose(image)


image = load_image('images/zed_left_792.png')
# image = image[280:1060, 320:1060, :]
image = image[320:1300, 280:1060, :]
quick_pose(image)



renderer = PixelMaskRenderer(path_OBJ, viewport_size, y_fov, distance,
                             light, top_only, roll, shift)
renderer.scene.ambient_light = [1.0, 1.0, 1.0]

inputs_to_shape = {'input_1': [H, W, num_channels]}
labels_to_shape = {'masks': [H, W, 4]}

processor = DomainRandomization(
    renderer, image_shape, image_paths, inputs_to_shape,
    labels_to_shape, num_occlusions)

for _ in range(100):
    sample = processor()
    image = sample['inputs']['input_1']
    masks = sample['labels']['masks']
    image = (image * 255).astype('uint8')
    # image, alpha, RGB_mask = renderer.render()
    # show_image((image * 255).astype('uint8'))
    quick_pose(image)
    # show_image(images)
