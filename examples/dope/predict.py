import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
from tqdm import tqdm

from tensorflow.keras.models import load_model

from paz.abstract import SequentialProcessor
from paz.pipelines import RandomizeRenderedImage
from paz import processors as pr
from paz.backend.camera import Camera
from paz.backend.quaternion import quarternion_to_rotation_matrix
from paz.backend.image.draw import draw_dot, draw_cube
from paz.abstract import GeneratingSequence
from paz.processors.draw import DrawBoxes3D
from paz.evaluation import evaluateIoU, evaluateADD, evaluateMSSD, evaluateMSPD

from scenes import SingleView
from pipelines import ImageGenerator

from model import custom_mse

np.set_printoptions(suppress=True)

description = 'Script for making a prediction using the DOPE model'
root_path = os.path.join(os.path.expanduser('~'), '.keras/paz/')
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-op', '--obj_path', nargs='+', type=str, help='Paths of 3D OBJ models',
                    default=os.path.join(
                        root_path,
                        'datasets/ycb/models/035_power_drill/textured.obj'))
parser.add_argument('-mp', '--model_path', type=str, help='Path of the TensorFlow model',
                    default=os.path.join(
                        root_path,
                        'datasets/ycb/models/035_power_drill/textured.obj'))
parser.add_argument('-id', '--background_images', type=str,
                    help='Path to directory containing background images',
                    default=None)
parser.add_argument('-ld', '--image_size', default=128, type=int,
                    help='Size of the side of a square image e.g. 64')
parser.add_argument('-sh', '--top_only', default=0, choices=[0, 1], type=int,
                    help='Flag for full sphere or top half for rendering')
parser.add_argument('-r', '--roll', default=3.14159, type=float,
                    help='Threshold for camera roll in radians')
parser.add_argument('-s', '--shift', default=0.05, type=float,
                    help='Threshold of random shift of camera')
parser.add_argument('-d', '--depth', nargs='+', type=float,
                    default=[0.5, 1.0],
                    help='Distance from camera to origin in meters')
parser.add_argument('-fv', '--y_fov', default=3.14159 / 4.0, type=float,
                    help='Field of view angle in radians')
parser.add_argument('-l', '--light', nargs='+', type=float,
                    default=[.5, 30],
                    help='Light intensity from poseur')
parser.add_argument('-sf', '--scaling_factor', default=8.0, type=float,
                    help='Downscaling factor of the images')
parser.add_argument('-ns', '--num_stages', default=6, type=int,
                    help='Number of stages for DOPE')

args = parser.parse_args()


def plot_belief_maps(image_original, real_belief_maps, predicted_belief_maps):
    num_rows = 3
    num_cols = 9
    fig, ax = plt.subplots(num_rows, num_cols)
    fig.set_size_inches(12, 10)

    ax[0, 0].imshow(image_original)
    for i in range(1, num_cols):
        ax[0, i].axis('off')

    for i in range(num_rows):
        for j in range(num_cols):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)

    # Show real belief maps
    for i in range(num_cols):
        ax[1, i].imshow(real_belief_maps[i], cmap='gray', vmin=0.0, vmax=1.0)

    # Show the predicted belief maps
    for i in range(num_cols):
        ax[2, i].imshow(predicted_belief_maps[i], cmap='gray', vmin=0.0, vmax=1.0)

    # plt.tight_layout()
    #fig.subplots_adjust(hspace=0.5)
    plt.show()


def initialize_values(model, renderer, batch_size=16):
    # Create a paz camera object
    camera = Camera()

    # focal_length = renderer.camera.get_projection_matrix()[0, 0]
    focal_length = 480#args.image_size
    image_center = (args.image_size / 2.0, args.image_size / 2.0)

    # building camera parameters
    camera.distortion = np.zeros((4, 1))
    camera.intrinsics = np.array([[focal_length, 0, image_center[0]],
                                  [0, focal_length, image_center[1]],
                                  [0, 0, 1]])

    # Make predictions
    image_paths = glob.glob(os.path.join(args.background_images, '*.jpg'))
    processor = ImageGenerator(renderer, args.image_size, int(args.image_size / args.scaling_factor), image_paths, num_occlusions=0, num_stages=args.num_stages)
    sequence = GeneratingSequence(processor, batch_size, 1)

    sequence_iterator = sequence.__iter__()
    batch = next(sequence_iterator)
    predictions = model.predict(batch[0]['input_1'])

    original_images = batch[0]['input_1']
    belief_maps = predictions[-1]

    #belief_maps = batch[1]['belief_maps_stage_6']
    belief_maps = belief_maps.transpose(0, 3, 1, 2)

    return camera, original_images, belief_maps, renderer.camera_rotations


def predict_pose(camera, original_image, belief_maps, camera_rotation, object_translation, bounding_box_points_3d):

    # Get the pixel positions of the belief maps
    predicted_bounding_box_pixels = list()
    for belief_map in belief_maps:
        # Normalize belief map
        belief_map /= np.sum(belief_map)

        center = np.where(belief_map == belief_map.max())
        x_center = center[1][0] * args.scaling_factor
        y_center = center[0][0] * args.scaling_factor

        predicted_bounding_box_pixels.append([x_center, y_center])

    predicted_bounding_box_pixels = np.asarray(predicted_bounding_box_pixels).astype(int)

    solve_PNP = pr.SolvePNP(bounding_box_points_3d, camera)
    pose6D_predicted = solve_PNP(predicted_bounding_box_pixels)

    # Make changes to have the same format as the real rotation
    predicted_rotation_matrix = quarternion_to_rotation_matrix(pose6D_predicted.quaternion)

    real_rotation_matrix = camera_rotation[:3, :3]
    real_rotation_matrix[[1, 2]] = -real_rotation_matrix[[1, 2]]
    real_translation = object_translation

    return real_rotation_matrix, real_translation, predicted_rotation_matrix, pose6D_predicted.translation


def plot_predictions(model, renderer):
    images_bounding_boxes = list()
    object_extent = renderer.meshes_original[0].mesh.extents/2
    bounding_box_points_3d = np.array([[0., 0., 0.],
                                       [object_extent[0], object_extent[1], object_extent[2]],
                                       [object_extent[0], object_extent[1], -object_extent[2]],
                                       [object_extent[0], -object_extent[1], object_extent[2]],
                                       [object_extent[0], -object_extent[1], -object_extent[2]],
                                       [-object_extent[0], object_extent[1], object_extent[2]],
                                       [-object_extent[0], object_extent[1], -object_extent[2]],
                                       [-object_extent[0], -object_extent[1], object_extent[2]],
                                       [-object_extent[0], -object_extent[1], -object_extent[2]]])

    camera, original_images, belief_maps_batch, camera_rotations = initialize_values(model, renderer)

    for original_image, belief_maps, camera_rotation, object_translation in zip(original_images, belief_maps_batch, camera_rotations, renderer.object_translations):
        # Make the predictions
        real_rotation_matrix, real_translation, predicted_rotation_matrix, predicted_translation = predict_pose(camera, original_image, belief_maps, camera_rotation, object_translation, bounding_box_points_3d)

        print("Real translation: {}".format(real_translation))
        print("Predicted translation: {}".format(predicted_translation))

        world_to_camera_rotation_vector_real, _ = cv2.Rodrigues(real_rotation_matrix)
        world_to_camera_rotation_vector_predicted, _ = cv2.Rodrigues(predicted_rotation_matrix)

        points2D_real, _ = cv2.projectPoints(bounding_box_points_3d, world_to_camera_rotation_vector_real, real_translation, camera.intrinsics, camera.distortion)
        points2D_predicted, _ = cv2.projectPoints(bounding_box_points_3d, world_to_camera_rotation_vector_predicted, predicted_translation, camera.intrinsics, camera.distortion)

        image_bounding_box = original_image * 255
        points2D_real_cube = points2D_real[[3, 4, 8, 7, 1, 2, 6, 5]]
        points2D_predicted_cube = points2D_predicted[[3, 4, 8, 7, 1, 2, 6, 5]]

        image_bounding_box = draw_cube(image_bounding_box.astype("uint8"), points2D_real_cube.astype(int), radius=2, thickness=2, color=(0, 255, 0))
        image_bounding_box = draw_cube(image_bounding_box.astype("uint8"), points2D_predicted_cube.astype(int), radius=2, thickness=2, color=(255, 0, 0))

        plt.imshow(image_bounding_box)
        plt.show()


def make_multiple_predictions(num_predictions, model, renderer):
    add_values = list()
    for i in tqdm(range(num_predictions)):
        add_value, image_original = make_single_prediction(model, renderer, plot=False)
        add_values.append(add_value)

        #print(add_value)
        #plt.imshow(image_original)
        #plt.show()

    print(sorted(add_values))

def make_single_prediction(model, renderer, plot=True):
    # Known bounding box points for the drill
    bounding_box_points_3d = np.array([[[0.0, 0.0, 0.0],
                                       [0.08157125,  0.0609625,  0.089668585],
                                       [0.08157125,  0.0609625,  -0.089668585],
                                       [0.08157125,  -0.0609625,  0.089668585],
                                       [0.08157125,  -0.0609625,  -0.089668585],
                                       [-0.08157125, 0.0609625, 0.089668585],
                                       [-0.08157125, 0.0609625, -0.089668585],
                                       [-0.08157125, -0.0609625, 0.089668585],
                                       [-0.08157125, -0.0609625, -0.089668585]]])

    extent_drill = [0.121925/2., 0.1631425/2., 0.17933717/2]

    # Generate image
    image_original, alpha_original, bounding_box_points, belief_maps, _, bounding_box_points_3d_real = renderer.render()

    real_bounding_box_points = bounding_box_points[0]

    # Prepare image for the network
    image_paths = glob.glob(os.path.join(args.images_directory, '*.jpg'))
    augment = RandomizeRenderedImage(image_paths, 0)
    image_original = augment(image_original, alpha_original)

    preprocessors_input = [pr.NormalizeImage()]
    preprocess_input = SequentialProcessor(preprocessors_input)
    preprocessed_image = preprocess_input(image_original)
    preprocessed_image = np.expand_dims(preprocessed_image, 0)

    # Feed the image into the network
    prediction = model.predict(preprocessed_image)

    # Get prediction of the last layer
    predicted_belief_maps = prediction[-1][0]

    # Transpose belief maps
    predicted_belief_maps = np.transpose(predicted_belief_maps, [2, 0, 1])

    if plot:
        plot_belief_maps(image_original, belief_maps[0], predicted_belief_maps)

    predicted_bounding_box_points = list()

    for belief_map in predicted_belief_maps:
        # Normalize belief map
        belief_map /= np.sum(belief_map)

        center = np.where(belief_map == belief_map.max())
        x_center = center[1][0]*args.scaling_factor
        y_center = center[0][0]*args.scaling_factor

        predicted_bounding_box_points.append([x_center, y_center])

    predicted_bounding_box_points = np.asarray(predicted_bounding_box_points)

    if plot:
        print("Predicted bounding box points: " + str(predicted_bounding_box_points))
        print("Real bounding box points: " + str(bounding_box_points))

    # Create a paz camera object
    camera = Camera()

    #focal_length = renderer.camera.get_projection_matrix()[0, 0]
    focal_length = image_size[1]
    image_center = (image_size[1] / 2.0, image_size[0] / 2.0)

    # building camera parameters
    camera.distortion = np.zeros((4, 1))
    camera.intrinsics = np.array([[focal_length, 0, image_center[0]],
                                  [0, focal_length, image_center[1]],
                                  [0, 0, 1]])

    # Perform the PnP algorithm
    solve_PNP = pr.SolvePNP(bounding_box_points_3d[0], camera)

    pose6D_real = solve_PNP(real_bounding_box_points)
    pose6D_predicted = solve_PNP(predicted_bounding_box_points)

    drawBoxes3D = DrawBoxes3D(camera, {None: extent_drill})
    image_bounding_boxes = drawBoxes3D(image_original.astype("uint8"), pose6D_real)

    # Color the bounding box points in the image
    circle_size = 2
    image_original_pil = Image.fromarray(image_bounding_boxes)
    draw = ImageDraw.Draw(image_original_pil)

    for bounding_box_point in real_bounding_box_points:
        draw.ellipse((bounding_box_point[0] - circle_size, bounding_box_point[1] - circle_size,
                      bounding_box_point[0] + circle_size, bounding_box_point[1] + circle_size,),
                     fill='blue', outline='blue')

    for center in predicted_bounding_box_points:
        draw.ellipse((center[0] - circle_size, center[1] - circle_size,
                      center[0] + circle_size, center[1] + circle_size,),
                     fill='yellow', outline='yellow')

    #if plot:
    print("Real pose: " + str(pose6D_real))
    print("Predicted pose: " + str(pose6D_predicted))
    print("Predicted rotation matrix: {}".format(quarternion_to_rotation_matrix(pose6D_predicted.quaternion)))
    print("Predicted translation: {}".format(pose6D_predicted.translation))

    print("Real points: " + str(real_bounding_box_points))
    print("Predicted points: " + str(predicted_bounding_box_points))

    real_bounding_box_points_3d = np.asarray([quarternion_to_rotation_matrix(pose6D_real.quaternion)@bb_point + np.squeeze(pose6D_real.translation) for bb_point in bounding_box_points_3d[0]])
    predicted_bounding_box_points_3d = np.asarray([quarternion_to_rotation_matrix(pose6D_predicted.quaternion)@bb_point + np.squeeze(pose6D_predicted.translation) for bb_point in bounding_box_points_3d[0]])

    iou_value = evaluateIoU(real_bounding_box_points_3d, predicted_bounding_box_points_3d,
                            pose6D_real, pose6D_predicted, np.asarray([0.1631425/2., 0.121925/2., 0.17933717/2]), num_sampled_points=1000)
    if plot:
        print("IoU: " + str(iou_value))

    add_value = evaluateADD(real_bounding_box_points_3d, predicted_bounding_box_points_3d)
    if plot:
        print("ADD: " + str(add_value))

    mssd_value = evaluateMSSD(real_bounding_box_points_3d, predicted_bounding_box_points_3d)
    if plot:
        print("MSSD: " + str(mssd_value))

    mspd_value = evaluateMSPD(real_bounding_box_points, predicted_bounding_box_points, renderer.viewport_size)
    if plot:
        print("MSPD: " + str(mspd_value))

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-.5, .5)
        ax.set_ylim(-.5, .5)
        ax.set_zlim(-.5, .5)
        ax.scatter(real_bounding_box_points_3d[:, 0], real_bounding_box_points_3d[:, 1], real_bounding_box_points_3d[:, 2])
        ax.scatter(predicted_bounding_box_points_3d[:, 0], predicted_bounding_box_points_3d[:, 1], predicted_bounding_box_points_3d[:, 2])
        plt.show()

    return add_value, image_original


if __name__ == "__main__":
    model = load_model(args.model_path, custom_objects={'custom_mse': custom_mse })
    colors = [np.array([255, 0, 0]), np.array([0, 255, 0])]
    renderer = SingleView(filepath=args.obj_path, colors=colors, viewport_size=(args.image_size, args.image_size),
                          y_fov=args.y_fov, distance=args.depth, light_bounds=args.light, top_only=bool(args.top_only),
                          roll=None, shift=None)

    #make_single_prediction(model, renderer, plot=True)
    #make_multiple_predictions(50, model, renderer)

    plot_predictions(model, renderer)
