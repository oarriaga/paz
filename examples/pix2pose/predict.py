import argparse
import os
import glob
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
import trimesh
import pyprogressivex

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
from paz.abstract.messages import Box2D, Pose6D

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


def initialize_values(renderer, model, batch_size=16):
    # Create a paz camera object
    camera = Camera()

    focal_length = 179  # image_size[1]
    image_center = (image_size[1] / 2.0, image_size[0] / 2.0)

    # building camera parameters
    camera.distortion = np.zeros((4, 1))
    camera.intrinsics = np.array([[focal_length, 0, image_center[0]],
                                  [0, focal_length, image_center[1]],
                                  [0, 0, 1]])

    rotation_matrices = [np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])]
    image_paths = glob.glob(os.path.join(args.images_directory, '*.jpg'))
    processor = DepthImageGenerator(renderer, image_size[0], image_paths, num_occlusions=0)
    sequence = GeneratingSequencePix2Pose(processor, model, batch_size, 1, rotation_matrices=rotation_matrices)

    sequence_iterator = sequence.__iter__()
    batch = next(sequence_iterator)
    predictions = model.predict(batch[0]['input_image'])

    # Get all the necessary images
    original_images = (batch[0]['input_image'] * 255).astype(np.int)
    color_images = ((batch[1]['color_output'] + 1) * 127.5).astype(np.int)
    predictions['color_output'] = ((predictions['color_output'] + 1) * 127.5).astype(np.int)

    return camera, original_images, color_images, predictions['color_output'], predictions['error_output'], renderer.object_rotations, renderer.camera_translations


def predict_pose(camera, color_image, error_image, real_rotation, camera_translation):
    # Just take the pixels from the color image where the error is not too high
    error_threshold = 0.15
    error_mask = np.ones_like(error_image)
    error_mask[error_image > error_threshold] = 0
    predicted_color_image = (color_image * error_mask).astype(int)

    # Get the pose from the predicted color image
    predicted_non_zero_pixels = np.argwhere(np.sum(predicted_color_image, axis=2))
    predicted_color_points = predicted_color_image[predicted_non_zero_pixels[:, 0], predicted_non_zero_pixels[:, 1]]
    solve_PNP = pr.SolvePNP((predicted_color_points / 127.5 - 1) * np.array([0.2, 0.15, 0.1]) / 2, camera)
    pose6D_predicted = solve_PNP(predicted_non_zero_pixels)

    # Transform the predicted rotation matrix to have the same format as the real one
    predicted_rotation_matrix = quarternion_to_rotation_matrix(pose6D_predicted.quaternion)
    predicted_rotation_matrix = predicted_rotation_matrix[[1, 0, 2]]
    predicted_rotation_matrix *= np.array([[1., -1., -1.], [1., -1., -1], [-1., 1., 1.]])
    predicted_translation = pose6D_predicted.translation

    # Somehow the rotations have a different sign when using predictPoints
    object_rotation = real_rotation * np.array([-1, 1, 1, 1])
    real_rotation_matrix = quarternion_to_rotation_matrix(object_rotation)

    real_translation = camera_translation
    real_translation = real_translation[[1, 0, 2]]
    real_translation[0] = -real_translation[0]
    print("Predicted translation: {}".format(pose6D_predicted.translation))

    return real_rotation_matrix, real_translation, predicted_rotation_matrix, predicted_translation


def predict_points(points_object_coords, rotation_matrix, translation, world_to_camera_rotation_vector, camera):
    # Calculate the real pixel locations of the bounding box points
    points3D = np.asarray([np.dot(rotation_matrix, point_object_coords) for point_object_coords in points_object_coords])
    # Axis are swapped on the predicted points
    points3D = points3D[:, [1, 0, 2]]
    points2D, _ = cv2.projectPoints(np.asarray(points3D), world_to_camera_rotation_vector,
                                         -np.squeeze(translation), camera.intrinsics, camera.distortion)
    # x- and y-coordinates are switched with the predicted points
    points2D = points2D[:, :, [1, 0]]
    return points2D, points3D


def plot_predictions(renderer, model):
    camera, original_images, _, predicted_color_images, predicted_error_images, object_rotations, camera_translations = initialize_values(renderer, model)
    images_bounding_boxes = list()

    for original_image, predicted_color_image, predicted_error_image, object_rotation, camera_translation in zip(original_images, predicted_color_images, predicted_error_images, object_rotations, camera_translations):
        # Make the predictions
        real_rotation_matrix, real_translation, predicted_rotation_matrix, predicted_translation = predict_pose(camera, predicted_color_image, predicted_error_image, object_rotation, camera_translation)
        print("Real rotation matrix: {}".format(real_rotation_matrix))
        print("Predicted rotation matrix: {}".format(predicted_rotation_matrix))

        # Define the bounding box points
        world_to_camera_rotation_vector, _ = cv2.Rodrigues(renderer.world_to_camera[:3, :3])
        object_extent = renderer.mesh_original.mesh.extents/2.
        bounding_box_points = [np.array([object_extent[0], -object_extent[1], object_extent[2]]),
                               np.array([object_extent[0], -object_extent[1], -object_extent[2]]),
                               np.array([-object_extent[0], -object_extent[1], -object_extent[2]]),
                               np.array([-object_extent[0], -object_extent[1], object_extent[2]]),
                               np.array([object_extent[0], object_extent[1], object_extent[2]]),
                               np.array([object_extent[0], object_extent[1], -object_extent[2]]),
                               np.array([-object_extent[0], object_extent[1], -object_extent[2]]),
                               np.array([-object_extent[0], object_extent[1], object_extent[2]])]

        # Get the 2D and 3D bounding box positions from rotation and translation
        points2D_real, points3D_real = predict_points(bounding_box_points, real_rotation_matrix, real_translation, world_to_camera_rotation_vector, camera)
        points2D_predicted, points3D_predicted = predict_points(bounding_box_points, predicted_rotation_matrix, predicted_translation, world_to_camera_rotation_vector, camera)

        # Draw the real and predicted cube
        image_bounding_boxes = draw_cube(original_image.astype("uint8"), points2D_predicted.astype(int), radius=1, thickness=1, color=(255, 0, 0))
        image_bounding_boxes = draw_cube(image_bounding_boxes.astype("uint8"), points2D_real.astype(int), radius=1, thickness=1, color=(0, 255, 0))

        images_bounding_boxes.append(image_bounding_boxes)

    fig, axs = plt.subplots(4, 4)
    axs = axs.flatten()

    for i, ax in enumerate(axs):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.imshow(images_bounding_boxes[i])

    plt.show()


def plot_predictions_custom_coloring(renderer):
    camera, images_original, real_poses, predicted_poses = test_custom_coloring(renderer)
    images_bounding_boxes = list()

    for image_original, real_pose, predicted_pose in zip(images_original, real_poses, predicted_poses):

        # Define the bounding box points
        world_to_camera_rotation_vector, _ = cv2.Rodrigues(renderer.world_to_camera[:3, :3])
        object_extent = renderer.mesh_original.mesh.extents/2.
        bounding_box_points = [np.array([object_extent[0], -object_extent[1], object_extent[2]]),
                               np.array([object_extent[0], -object_extent[1], -object_extent[2]]),
                               np.array([-object_extent[0], -object_extent[1], -object_extent[2]]),
                               np.array([-object_extent[0], -object_extent[1], object_extent[2]]),
                               np.array([object_extent[0], object_extent[1], object_extent[2]]),
                               np.array([object_extent[0], object_extent[1], -object_extent[2]]),
                               np.array([-object_extent[0], object_extent[1], -object_extent[2]]),
                               np.array([-object_extent[0], object_extent[1], object_extent[2]])]

        # Get the 2D and 3D bounding box positions from rotation and translation
        translation = np.array([0., 0., -0.4])
        points2D_real, points3D_real = predict_points(bounding_box_points, real_pose, translation, world_to_camera_rotation_vector, camera)
        points2D_predicted, points3D_predicted = predict_points(bounding_box_points, predicted_pose, translation, world_to_camera_rotation_vector, camera)

        # Draw the real and predicted cube
        image_bounding_boxes = draw_cube(image_original.astype("uint8"), points2D_predicted.astype(int), radius=1, thickness=1, color=(255, 0, 0))
        image_bounding_boxes = draw_cube(image_bounding_boxes.astype("uint8"), points2D_real.astype(int), radius=1, thickness=1, color=(0, 255, 0))

        images_bounding_boxes.append(image_bounding_boxes)

    fig, axs = plt.subplots(2, 2)
    axs = axs.flatten()

    for i, ax in enumerate(axs):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.imshow(images_bounding_boxes[i])

    plt.show()


def calculate_error(renderer, model, rotation_matrices):
    world_to_camera_rotation_vector, _ = cv2.Rodrigues(renderer.world_to_camera[:3, :3])
    camera, original_images, _, predicted_color_images, predicted_error_images, object_rotations, camera_translations = initialize_values(renderer, model, batch_size=50)
    points3D_object_coords = renderer.mesh_original.mesh.primitives[0].positions

    add_values = list()

    for original_image, predicted_color_image, predicted_error_image, object_rotation, camera_translation in zip(original_images, predicted_color_images, predicted_error_images, object_rotations, camera_translations):
        real_rotation_matrix, real_translation, predicted_rotation_matrix, predicted_translation = predict_pose(camera, predicted_color_image, predicted_error_image, object_rotation, camera_translation)

        print("Real rotation matrix: {}".format(real_rotation_matrix))
        print("Predicted rotation matrix: {}".format(predicted_rotation_matrix))

        # Iterate over all rotation matrices to find the smallest possible error
        min_add_value = np.iinfo(np.uint64).max
        for rotation_matrix in rotation_matrices:
            real_points2D, real_points3D = predict_points(points3D_object_coords, real_rotation_matrix, real_translation, world_to_camera_rotation_vector, camera)

            predicted_rotation_matrix = np.dot(predicted_rotation_matrix, rotation_matrix)
            predicted_points2D, predicted_points3D = predict_points(points3D_object_coords, predicted_rotation_matrix, predicted_translation, world_to_camera_rotation_vector, camera)

            add_value = evaluateADD(real_points3D, predicted_points3D)
            if add_value < min_add_value:
                min_add_value = add_value

        add_values.append(min_add_value)

    print("ADD values: {}".format(sorted(add_values)))


def test_custom_coloring(renderer, batch_size=4):
    #renderer = SingleView(filepath=args.obj_path,
    #                      filepath_half_object="/home/fabian/.keras/datasets/custom_objects/symmetric_object_half.obj",
    #                      viewport_size=image_size, y_fov=args.y_fov, distance=args.depth, light_bounds=args.light,
    #                      top_only=bool(args.top_only),
    #                      roll=None, shift=None)

    focal_length = 179  # image_size[1]
    image_center = (image_size[1] / 2.0, image_size[0] / 2.0)

    # building camera parameters
    intrinsics = np.array([[focal_length, 0, image_center[0]],
                            [0, focal_length, image_center[1]],
                            [0, 0, 1]])

    camera = Camera()

    focal_length = 179  # image_size[1]
    image_center = (image_size[1] / 2.0, image_size[0] / 2.0)

    # building camera parameters
    camera.distortion = np.zeros((4, 1))
    camera.intrinsics = np.array([[focal_length, 0, image_center[0]],
                                  [0, focal_length, image_center[1]],
                                  [0, 0, 1]])

    real_poses, predicted_poses, images_original = list(), list(), list()

    for i in range(batch_size):
        image_original, image_colors, alpha_original = renderer.render_custom_coloring()
        images_original.append(image_original)

        #plt.imshow(image_colors)
        #plt.show()

        predicted_non_zero_pixels = np.argwhere(np.sum(image_colors, axis=2))
        predicted_color_points = image_colors[predicted_non_zero_pixels[:, 0], predicted_non_zero_pixels[:, 1]]
        predicted_color_points = (predicted_color_points / 127.5 - 1) * np.array([0.2, 0.15, 0.1]) / 2

        # Add the rotated color points
        rotation_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        predicted_color_points_rotated = [np.dot(rotation_matrix, color_point) for color_point in predicted_color_points]

        #predicted_non_zero_pixels = np.concatenate((predicted_non_zero_pixels, predicted_non_zero_pixels))
        #predicted_color_points = np.concatenate((predicted_color_points, predicted_color_points_rotated))

        solve_PNP = pr.SolvePNP(predicted_color_points, camera)
        pose6D_predicted = solve_PNP(predicted_non_zero_pixels)
        predicted_rot_matrix = quarternion_to_rotation_matrix(pose6D_predicted.quaternion)
        predicted_rot_matrix = predicted_rot_matrix[[1, 0, 2]]
        predicted_rot_matrix[0] = -predicted_rot_matrix[0]

        threshold = 8.0

        poses, labeling = pyprogressivex.find6DPoses(np.ascontiguousarray(predicted_non_zero_pixels), np.ascontiguousarray(predicted_color_points),
                                                     np.ascontiguousarray(intrinsics), threshold, maximum_model_number=1, neighborhood_ball_radius=30)

        print("poses: {}".format(poses))

        pose01 = poses[:3, :3]
        pose01 = pose01[[1, 0, 2]]
        pose01[0] = -pose01[0]

        real_rotation = renderer.rotation_inverse*np.array([-1, 1, 1, 1])
        real_poses.append(quarternion_to_rotation_matrix(real_rotation))
        predicted_poses.append(pose01)

        predicted_pose = trimesh.transformations.quaternion_from_matrix(pose01)
        predicted_pose = -predicted_pose
        predicted_pose[-1] = -predicted_pose[-1]
        predicted_pose = predicted_pose * np.array([-1, 1, 1, 1])

        print("Euler rotation predicted: {}".format(trimesh.transformations.euler_from_matrix(quarternion_to_rotation_matrix(predicted_pose))))
        print("Euler rotation true: {}".format(trimesh.transformations.euler_from_matrix(quarternion_to_rotation_matrix(real_rotation))))

    return camera, images_original, real_poses, predicted_poses


if __name__ == "__main__":
    rotation_matrices = [np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])]
    model = load_model(args.model_path, custom_objects={'loss_color_unwrapped': loss_color_wrapped(rotation_matrices), 'loss_error': loss_error})
    renderer = SingleView(filepath=args.obj_path, filepath_half_object="/home/fabian/.keras/datasets/custom_objects/symmetric_object_half.obj",
                          viewport_size=image_size, y_fov=args.y_fov, distance=args.depth, light_bounds=args.light, top_only=bool(args.top_only),
                          roll=None, shift=None)

    rotation_matrices_error = [np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]), np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])]
    #plot_predictions(renderer, model)
    calculate_error(renderer, model, rotation_matrices_error)

    #renderer = SingleView(filepath=args.obj_path,
    #                      filepath_half_object="/home/fabian/.keras/datasets/custom_objects/symmetric_object_half.obj",
    #                      viewport_size=image_size, y_fov=args.y_fov, distance=args.depth, light_bounds=args.light,
    #                      top_only=bool(args.top_only),
    #                      roll=None, shift=None)
    #plot_predictions_custom_coloring(renderer)


"""def make_single_prediction_old(model, renderer, plot=True, rotation_matrices=None):
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
    print("Real pose: {}".format(pose6D_real))

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

    print("Predicted pose: {}".format(pose6D_predicted))
    print("Predicted rotation matrix: {}".format(predicted_rotation_matrix))

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

    iou_value = evaluateIoU(pose6D_real, pose6D_predicted, np.asarray([0.2/2., 0.15/2., 0.1/2]), num_sampled_points=1000)
    print("IoU: " + str(iou_value))

    add_value = evaluateADD(real_points_3d, predicted_points_3d)
    print("ADD: " + str(add_value))

    mssd_value = evaluateMSSD(real_points_3d, predicted_points_3d)
    print("MSSD: " + str(mssd_value))

    #mspd_value = evaluateMSPD(points2D_real, points2D_predicted, renderer.viewport_size)
    #print("MSPD: " + str(mspd_value))
"""
