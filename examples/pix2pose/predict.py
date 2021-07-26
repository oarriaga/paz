import argparse
import os
import glob

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
import trimesh

from tensorflow.keras.models import load_model

from paz.abstract import SequentialProcessor
from paz import processors as pr
from paz.backend.camera import Camera
from paz.backend.quaternion import quarternion_to_rotation_matrix, rotation_vector_to_quaternion
from paz.processors.draw import DrawBoxes3D
from paz.evaluation import evaluateIoU, evaluateADD, evaluateMSSD, evaluateMSPD
from paz.abstract.sequence import GeneratingSequencePix2Pose
from paz.backend.image.draw import draw_dot, draw_cube
from paz.backend.keypoints import project_points3D

from pipelines import DepthImageGenerator, RendererDataGenerator, make_batch_discriminator
from scenes import SingleView
from model import loss_color_wrapped, loss_error



description = 'Script for making a prediction using the pix2pose model'
root_path = os.path.join(os.path.expanduser('~'), '.keras/paz/')
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-op', '--obj_path', type=str, help='Paths of 3D OBJ models',
                    default=os.path.join(
                        root_path,
                        'datasets/ycb/models/035_power_drill/textured.obj'))
parser.add_argument('-mp', '--model_path', type=str, help='Path of the TensorFlow model',
                    default=os.path.join(
                        root_path,
                        'datasets/ycb/models/035_power_drill/textured.obj'))
parser.add_argument('-ld', '--image_size', default=128, type=int, nargs='+',
                    help='Size of the side of a square image e.g. 64')
parser.add_argument('-id', '--images_directory', type=str,
                    help='Path to directory containing background images',
                    default="/media/fabian/Data/Masterarbeit/data/VOCdevkit/VOC2012/JPEGImages")
parser.add_argument('-sh', '--top_only', default=0, choices=[0, 1], type=int,
                    help='Flag for full sphere or top half for rendering')
parser.add_argument('-r', '--roll', default=3.14159, type=float,
                    help='Threshold for camera roll in radians')
parser.add_argument('-s', '--shift', default=0.05, type=float,
                    help='Threshold of random shift of camera')
parser.add_argument('-d', '--depth', nargs='+', type=float,
                    default=[0.3, 0.5],
                    help='Distance from camera to origin in meters')
parser.add_argument('-fv', '--y_fov', default=3.14159 / 4.0, type=float,
                    help='Field of view angle in radians')
parser.add_argument('-l', '--light', nargs='+', type=float,
                    default=[.5, 30],
                    help='Light intensity from poseur')
parser.add_argument('-sf', '--scaling_factor', default=8.0, type=float,
                    help='Downscaling factor of the images')

args = parser.parse_args()
image_size = tuple(args.image_size)


def make_single_prediction(model, renderer, plot=True, rotation_matrices=None):
    # Prepare image for the network
    image_paths = glob.glob(os.path.join(args.images_directory, '*.jpg'))
    processor = DepthImageGenerator(renderer, image_size[0], image_paths, num_occlusions=0)
    sequence = GeneratingSequencePix2Pose(processor, model, 4, 2, rotation_matrices=rotation_matrices)

    sequence_iterator = sequence.__iter__()
    batch = next(sequence_iterator)
    predictions = model.predict(batch[0]['input_image'])

    original_images = (batch[0]['input_image'] * 255).astype(np.int)
    color_images = ((batch[1]['color_output'] + 1) * 127.5).astype(np.int)
    predictions['color_output'] = ((predictions['color_output'] + 1) * 127.5).astype(np.int)
    #predictions['error_output'] = ((predictions['error_output'] + 1) * 127.5).astype(np.int)
    # color_images = batch[1]['color_output']

    if plot:
        fig, ax = plt.subplots(4, 4)
        cols = ["Input image", "Ground truth", "Predicted image", "Predicted error"]

        for i in range(4):
            ax[0, i].set_title(cols[i])
            for j in range(4):
                ax[i, j].get_xaxis().set_visible(False)
                ax[i, j].get_yaxis().set_visible(False)

        for i in range(4):
            ax[i, 0].imshow(original_images[i])
            ax[i, 1].imshow(color_images[i])
            ax[i, 2].imshow(predictions['color_output'][i])
            ax[i, 3].imshow(np.squeeze(predictions['error_output'][i]))

        plt.tight_layout()
        plt.show()

    # Create a paz camera object
    camera = Camera()

    #focal_length02 = renderer.camera.get_projection_matrix()[0, 0]
    focal_length = 179#image_size[1]
    image_center = (image_size[1] / 2.0, image_size[0] / 2.0)

    # building camera parameters
    camera.distortion = np.zeros((4, 1))
    camera.intrinsics = np.array([[focal_length, 0, image_center[0]],
                                  [0, focal_length, image_center[1]],
                                  [0, 0, 1]])

    # Calculate pose for one image
    real_color_image = color_images[-1]
    predicted_color_image = predictions['color_output'][-1]
    predicted_error_image = predictions['error_output'][-1]

    plt.imshow(np.squeeze(predicted_error_image))
    plt.show()

    real_non_zero_pixels = np.argwhere(np.sum(real_color_image, axis=2))
    real_color_points = real_color_image[real_non_zero_pixels[:, 0], real_non_zero_pixels[:, 1]]

    # Perform the PnP algorithm on the real color image
    solve_PNP = pr.SolvePNP((real_color_points/127.5-1) * np.array([0.2, 0.15, 0.1])/2, camera)
    pose6D_real = solve_PNP(real_non_zero_pixels)
    real_pose_rotated = np.dot(np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]), quarternion_to_rotation_matrix(pose6D_real.quaternion)).T
    object_to_world = np.dot(quarternion_to_rotation_matrix(pose6D_real.quaternion), renderer.camera_to_world[:3, :3])
    print("Real pose: {}".format(pose6D_real))
    print("Real pose eul: {}".format(trimesh.transformations.euler_from_quaternion(pose6D_real.quaternion)))
    print("Real rotation matrix: {}".format(object_to_world))
    print("Real pose rotated: {}".format(real_pose_rotated))

    plt.imshow(np.abs(predicted_color_image.astype(float)/255. - real_color_image.astype(float)/255.))
    plt.show()

    # Pixels in the predicted color image that have an error that is too high (predicted error image)
    # are not taken into consideration for the pose prediction
    plt.imshow(predicted_color_image)
    plt.show()

    error_threshold = 0.15
    error_mask = np.ones_like(predicted_error_image)
    error_mask[predicted_error_image > error_threshold] = 0
    predicted_color_image = (predicted_color_image*error_mask).astype(int)

    plt.imshow(predicted_color_image)
    plt.show()

    # Get the pose from the predicted color image
    predicted_non_zero_pixels = np.argwhere(np.sum(predicted_color_image, axis=2))
    predicted_color_points = predicted_color_image[predicted_non_zero_pixels[:, 0], predicted_non_zero_pixels[:, 1]]
    solve_PNP = pr.SolvePNP((predicted_color_points/127.5-1) * np.array([0.2, 0.15, 0.1])/2, camera)
    pose6D_predicted = solve_PNP(predicted_non_zero_pixels)
    print("Predicted pose: {}".format(pose6D_predicted))
    print("Predicted rotation matrix: {}".format(quarternion_to_rotation_matrix(pose6D_predicted.quaternion)))

    # Points in object coordinates
    object_bounding_box_points = [np.array([0.1, -0.075, 0.05]), np.array([0.1, -0.075, -0.05]), np.array([-0.1, -0.075, -0.05]),
                                 np.array([-0.1, -0.075, 0.05]), np.array([0.1, 0.075, 0.05]), np.array([0.1, 0.075, -0.05]),
                                 np.array([-0.1, 0.075, -0.05]), np.array([-0.1, 0.075, 0.05])]

    # Transform the points to the world coordinate frame
    real_points_world_coords = list()
    predicted_points_world_coords = list()

    # True rotation of the object
    object_rotation = renderer.rotation_object
    # Somehow the rotations have a different sign when using predictPoints
    object_rotation *= np.array([-1, 1, 1, 1])
    real_rotation_matrix = quarternion_to_rotation_matrix(object_rotation)
    #object_rotation = object_rotation[[1, 0, 2, 3]]

    print("Another rotation matrix: {}".format(real_rotation_matrix))

    for bounding_box_point in object_bounding_box_points:
        real_points_world_coords.append(np.dot(real_rotation_matrix, bounding_box_point))

    # Transform the predicted rotation matrix to have the same format as the real one
    predicted_rotation_matrix = quarternion_to_rotation_matrix(pose6D_predicted.quaternion)
    predicted_rotation_matrix = predicted_rotation_matrix[[1, 0, 2]]
    predicted_rotation_matrix *= np.array([[1., -1., -1.], [1., -1., -1], [-1., 1., 1.]])
    for bounding_box_point in object_bounding_box_points:
        predicted_points_world_coords.append(np.dot(predicted_rotation_matrix, bounding_box_point))

    print("real_points_world_coords: {}".format(real_points_world_coords))
    print("predicted_points_world_coords: {}".format(predicted_points_world_coords))

    ## Plot the real bounding box
    # Transform the rotation matrix into a rotation vector
    world_to_camera_rotation_vector, _ = cv2.Rodrigues(renderer.world_to_camera[:3, :3])
    real_points_world_coords = np.asarray(real_points_world_coords)
    # Axis are swapped on the predicted points
    real_points_world_coords = real_points_world_coords[:, [1, 0, 2]]
    points2D_real, _ = cv2.projectPoints(np.asarray(real_points_world_coords), world_to_camera_rotation_vector,
                                         -np.squeeze(pose6D_real.translation), camera.intrinsics, camera.distortion)

    # x- and y-coordinates are switched with the predicted points
    points2D_real = points2D_real[:, :, [1, 0]]
    image_bounding_boxes = draw_cube(real_color_image.astype("uint8"), points2D_real.astype(int), radius=1, thickness=1)

    ## Plot the predicted bounding box
    # Transform the rotation matrix into a rotation vector
    world_to_camera_rotation_vector, _ = cv2.Rodrigues(renderer.world_to_camera[:3, :3])
    predicted_points_world_coords = np.asarray(predicted_points_world_coords)
    # Axis are swapped on the predicted points
    predicted_points_world_coords = predicted_points_world_coords[:, [1, 0, 2]]
    points2D_predicted, _ = cv2.projectPoints(np.asarray(predicted_points_world_coords), world_to_camera_rotation_vector,
                                         -np.squeeze(pose6D_predicted.translation), camera.intrinsics, camera.distortion)

    # x- and y-coordinates are switched with the predicted points
    points2D_predicted = points2D_predicted[:, :, [1, 0]]
    image_bounding_boxes = draw_cube(image_bounding_boxes.astype("uint8"), points2D_predicted.astype(int), radius=1, thickness=1, color=(255, 0, 0))

    plt.imshow(image_bounding_boxes)
    plt.show()

    mesh_points_3d = renderer.mesh_original.mesh.primitives[0].positions
    real_points_3d = np.asarray([real_rotation_matrix@point + np.squeeze(pose6D_real.translation) for point in mesh_points_3d])
    predicted_points_3d = np.asarray([predicted_rotation_matrix@point + np.squeeze(pose6D_predicted.translation) for point in mesh_points_3d])

    iou_value = evaluateIoU(real_points_3d, predicted_points_3d,
                            pose6D_real, pose6D_predicted, np.asarray([0.1631425/2., 0.121925/2., 0.17933717/2]), num_sampled_points=1000)
    print("IoU: " + str(iou_value))

    add_value = evaluateADD(real_points_3d, predicted_points_3d)
    print("ADD: " + str(add_value))

    mssd_value = evaluateMSSD(real_points_3d, predicted_points_3d)
    print("MSSD: " + str(mssd_value))

    #mspd_value = evaluateMSPD(points2D_real, points2D_predicted, renderer.viewport_size)
    #print("MSPD: " + str(mspd_value))


if __name__ == "__main__":
    rotation_matrices = [np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])]
    model = load_model(args.model_path, custom_objects={'loss_color_unwrapped': loss_color_wrapped(rotation_matrices), 'loss_error': loss_error})
    print(image_size)
    renderer = SingleView(filepath=args.obj_path, viewport_size=image_size,
                          y_fov=args.y_fov, distance=args.depth, light_bounds=args.light, top_only=bool(args.top_only),
                          roll=None, shift=None)

    make_single_prediction(model, renderer, plot=True, rotation_matrices=rotation_matrices)