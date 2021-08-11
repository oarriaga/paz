import os
import json
import argparse
import glob
import matplotlib.pyplot as plt
import cv2

import numpy as np
import trimesh
from tqdm import tqdm

from tensorflow.keras.utils import get_file
from tensorflow.keras.models import load_model, Model
from sklearn.metrics.pairwise import cosine_similarity as measure
from paz.backend.camera import VideoPlayer, Camera
from paz.processors import DrawBoxes3D, PlotErrorCurve
from paz.abstract.messages import Pose6D
from paz import processors as pr
from paz.evaluation import evaluateIoU, evaluateADD, evaluateMSSD, evaluateMSPD

from scenes import DictionaryView, SingleView

from model import AutoEncoder
from pipelines import ImplicitRotationPredictor, DomainRandomizationProcessor

parser = argparse.ArgumentParser(description='Implicit orientation demo')
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
parser.add_argument('-bi', '--background_images_directory', type=str,
                    help='Path to directory containing background images')
parser.add_argument('-f', '--y_fov', type=float, default=3.14159 / 4.0,
                    help='field of view')
parser.add_argument('-v', '--viewport_size', type=int, default=128,
                    help='Size of rendered images')
parser.add_argument('-d', '--distance', type=float, default=0.6,
                    help='Distance between camera and 3D model')
parser.add_argument('-s', '--shift', type=float, default=0.01,
                    help='Shift')
parser.add_argument('-l', '--light', type=int, default=10,
                    help='Light intensity')
parser.add_argument('-b', '--background', type=int, default=0,
                    help='Plain background color')
parser.add_argument('-r', '--roll', type=float, default=3.14159,
                    help='Maximum roll')
parser.add_argument('-t', '--translate', type=float, default=0.01,
                    help='Maximum translation')
parser.add_argument('-p', '--top_only', type=int, default=0,
                    help='Rendering mode')
parser.add_argument('--theta_steps', type=int, default=50,
                    help='Amount of steps taken in the X-Y plane')
parser.add_argument('--phi_steps', type=int, default=50,
                    help='Amount of steps taken from the Z-axis')
parser.add_argument('--weights_path', type=str, help='Path to trained model')
parser.add_argument('--obj_path', type=str, help='Path to .obj file')
parser.add_argument('-ld', '--latent_dimensions', type=int, default=128,
                    help='Number of latent dimensions')
args = parser.parse_args()


def initialize_values():
    obj_path = get_file(args.obj_path, None)

    dictionaryView = DictionaryView(
        obj_path, (args.viewport_size, args.viewport_size), args.y_fov,
        args.distance, bool(args.top_only), 5, args.theta_steps,
        args.phi_steps)

    encoder = AutoEncoder((args.viewport_size, args.viewport_size, 3), args.latent_dimensions, mode='encoder')
    encoder.load_weights(args.weights_path, by_name=True)
    decoder = AutoEncoder((args.viewport_size, args.viewport_size, 3), args.latent_dimensions, mode='decoder')
    decoder.load_weights(args.weights_path, by_name=True)

    inference = ImplicitRotationPredictor(encoder, decoder, measure, dictionaryView)

    background_image_paths = glob.glob(os.path.join(args.background_images_directory, '*.jpg'))
    renderer = SingleView(args.obj_path, (args.viewport_size, args.viewport_size),
                          args.y_fov, args.distance, [5, 5], bool(args.top_only),
                          None, None)
    processor = DomainRandomizationProcessor(renderer, background_image_paths, 0, split=pr.TEST)
    return processor, inference


def plot_predictions(num_predictions=4):
    processor, inference = initialize_values()
    mesh_extents = processor.renderer.mesh.mesh.extents
    mesh_extents = mesh_extents[[1, 0, 2]]

    focal_length = 75#args.viewport_size
    image_center = (args.viewport_size / 2.0, args.viewport_size / 2.0)
    camera = Camera()
    camera.distortion = np.zeros((4, 1))
    camera.intrinsics = np.array([[focal_length, 0, image_center[0]],
                                  [0, focal_length, image_center[1]],
                                  [0, 0, 1]])

    drawBoxes3D = DrawBoxes3D(camera, {None: mesh_extents}, thickness=1, radius=2)

    # Somehow something doesn't add up with the predicted rotation matrix
    # and projectPoints in the drawBox method, so we have to multiply
    # the rotation matrix with this mask
    rotation_matrix_mask = np.array([[1, -1, -1, -1],
                                     [-1, 1, 1, 1],
                                     [-1, 1, 1, 1],
                                     [1, 1, 1, 1]])

    images_bounding_boxes = list()

    for i in range(num_predictions**2):
        input_image, label_image, matrices = processor()
        inference_dict = inference(input_image)

        # Get real 6D pose
        real_rotation_matrix = matrices[0]
        real_rotation_matrix = np.reshape(real_rotation_matrix, (4, 4))
        real_rotation_matrix = np.multiply(real_rotation_matrix, rotation_matrix_mask)
        real_translation = real_rotation_matrix[:3, 3]
        real_rotation_vector, _ = cv2.Rodrigues(real_rotation_matrix[:3, :3])
        real_6Dpose = Pose6D.from_rotation_vector(real_rotation_vector, real_translation)

        # Get predicted 6D pose
        predicted_rotation_matrix = inference_dict['matrices'][0]
        predicted_rotation_matrix = np.reshape(predicted_rotation_matrix, (4, 4))
        predicted_rotation_matrix = np.multiply(predicted_rotation_matrix, rotation_matrix_mask)
        predicted_translation = predicted_rotation_matrix[:3, 3]
        predicted_rotation_vector, _ = cv2.Rodrigues(predicted_rotation_matrix[:3, :3])
        predicted_6Dpose = Pose6D.from_rotation_vector(predicted_rotation_vector, predicted_translation)

        image_3Dbox = drawBoxes3D(input_image, real_6Dpose, color=(0, 255, 0))
        image_3Dbox = drawBoxes3D(input_image, predicted_6Dpose, color=(255, 0, 0))
        images_bounding_boxes.append(image_3Dbox)


    fig, axs = plt.subplots(num_predictions, num_predictions)
    axs = axs.flatten()

    for i, ax in enumerate(axs):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.imshow(images_bounding_boxes[i])

    plt.show()


def calculate_error(num_samples=50):
    processor, inference = initialize_values()
    mesh_extents = processor.renderer.mesh.mesh.extents
    mesh_extents = mesh_extents[[1, 0, 2]]

    focal_length = 75  # args.viewport_size
    image_center = (args.viewport_size / 2.0, args.viewport_size / 2.0)
    camera = Camera()
    camera.distortion = np.zeros((4, 1))
    camera.intrinsics = np.array([[focal_length, 0, image_center[0]],
                                  [0, focal_length, image_center[1]],
                                  [0, 0, 1]])

    drawBoxes3D = DrawBoxes3D(camera, {None: mesh_extents}, thickness=1, radius=2)

    # Somehow something doesn't add up with the predicted rotation matrix
    # and projectPoints in the drawBox method, so we have to multiply
    # the rotation matrix with this mask
    rotation_matrix_mask = np.array([[1, -1, -1, -1],
                                     [-1, 1, 1, 1],
                                     [-1, 1, 1, 1],
                                     [1, 1, 1, 1]])

    images_bounding_boxes = list()
    add_values = list()
    mesh_points3D = processor.renderer.mesh.mesh.primitives[0].positions

    for i in tqdm(range(num_samples)):
        input_image, label_image, matrices = processor()
        inference_dict = inference(input_image)

        # Get real 6D pose
        real_rotation_matrix = matrices[0]
        real_rotation_matrix = np.reshape(real_rotation_matrix, (4, 4))
        real_rotation_matrix = np.multiply(real_rotation_matrix, rotation_matrix_mask)
        real_translation = real_rotation_matrix[:3, 3]


        # Get predicted 6D pose
        predicted_rotation_matrix = inference_dict['matrices'][0]
        predicted_rotation_matrix = np.reshape(predicted_rotation_matrix, (4, 4))
        predicted_rotation_matrix = np.multiply(predicted_rotation_matrix, rotation_matrix_mask)
        predicted_translation = predicted_rotation_matrix[:3, 3]

        real_points3D = mesh_points3D@real_rotation_matrix[:3, :3] + real_translation
        predicted_points3D = mesh_points3D@predicted_rotation_matrix[:3, :3] + predicted_translation

        add_value = evaluateADD(real_points3D, predicted_points3D)
        add_values.append(add_value)

        if add_value > 0.1:
            predicted_rotation_vector, _ = cv2.Rodrigues(predicted_rotation_matrix[:3, :3])
            predicted_6Dpose = Pose6D.from_rotation_vector(predicted_rotation_vector, predicted_translation)

            real_rotation_vector, _ = cv2.Rodrigues(real_rotation_matrix[:3, :3])
            real_6Dpose = Pose6D.from_rotation_vector(real_rotation_vector, real_translation)

            image_3Dbox = drawBoxes3D(input_image, real_6Dpose, color=(0, 255, 0))
            image_3Dbox = drawBoxes3D(input_image, predicted_6Dpose, color=(255, 0, 0))

            plt.imshow(image_3Dbox)
            plt.show()

    print(sorted(add_values))
    plotErrorCurve = PlotErrorCurve(max_error=0.15, num_steps=100)
    plotErrorCurve(add_values, title="ADD values implicit orientation", x_label="ADD values", y_label="Percentage of samples below error value")


if __name__ == "__main__":
    calculate_error(num_samples=1000)