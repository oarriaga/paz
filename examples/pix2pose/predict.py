import argparse
import os
import glob

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2

from tensorflow.keras.models import load_model

from paz.abstract import SequentialProcessor
from paz import processors as pr
from paz.backend.camera import Camera
from paz.backend.quaternion import quarternion_to_rotation_matrix
from paz.processors.draw import DrawBoxes3D
from paz.evaluation import evaluateIoU, evaluateADD, evaluateMSSD, evaluateMSPD
from paz.abstract.sequence import GeneratingSequencePix2Pose
from paz.backend.keypoints import project_points3D

from pipelines import DepthImageGenerator, RendererDataGenerator, make_batch_discriminator
from scenes import SingleView
from model import loss_color, loss_error



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

    #extent_drill = [0.121925/2., 0.1631425/2., 0.17933717/2.]
    extent_drill = [1. / 2., 1. / 2., 1. / 2.]

    # Prepare image for the network
    image_paths = glob.glob(os.path.join(args.images_directory, '*.jpg'))
    print(image_paths)
    processor = DepthImageGenerator(renderer, image_size[0], image_paths, num_occlusions=0)
    sequence = GeneratingSequencePix2Pose(processor, model, 16, 20)

    sequence_iterator = sequence.__iter__()
    batch = next(sequence_iterator)
    predictions = model.predict(batch[0]['input_image'])

    original_images = (batch[0]['input_image'] * 255).astype(np.int)
    color_images = ((batch[1]['color_output'] + 1) * 127.5).astype(np.int)
    predictions['color_output'] = ((predictions['color_output'] + 1) * 127.5).astype(np.int)
    predictions['error_output'] = ((predictions['error_output'] + 1) * 127.5).astype(np.int)
    # color_images = batch[1]['color_output']

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

    #focal_length = renderer.camera.get_projection_matrix()[0, 0]
    focal_length = image_size[1]
    image_center = (image_size[1] / 2.0, image_size[0] / 2.0)

    # building camera parameters
    camera.distortion = np.zeros((4, 1))
    camera.intrinsics = np.array([[focal_length, 0, image_center[0]],
                                  [0, focal_length, image_center[1]],
                                  [0, 0, 1]])

    # Find the non-zero pixels in the image
    real_non_zero_pixels = np.argwhere(np.sum(real_color_image, axis=2))
    real_color_points = real_color_image[real_non_zero_pixels[:, 0], real_non_zero_pixels[:, 1]]
    print("Real pixels: " + str(real_non_zero_pixels))

    # Perform the PnP algorithm
    solve_PNP = pr.SolvePNP(real_color_points/255., camera)

    pose6D_real = solve_PNP(real_non_zero_pixels)
    print(pose6D_real)
    #pose6D_predicted = solve_PNP(predicted_bounding_box_points)
    projected_pixels = np.squeeze(project_points3D(real_color_points/255., pose6D_real, camera))
    print("Projected pixels: " + str(projected_pixels))

    real_mask = np.zeros(image_size)
    real_mask[real_non_zero_pixels[:, 0], real_non_zero_pixels[:, 1]] = 1

    predicted_mask = np.zeros(image_size)
    predicted_mask[np.clip(projected_pixels[:, 0].astype(np.int), 0, image_size[0]), np.clip(projected_pixels[:, 1].astype(np.int), 0, image_size[1])] = 1

    plt.imshow(real_mask)
    plt.title("Real mask")
    plt.show()
    plt.imshow(predicted_mask)
    plt.title("Predicted mask")
    plt.show()

    drawBoxes3D = DrawBoxes3D(camera, {None: extent_drill})
    image_bounding_boxes = drawBoxes3D(real_color_image.astype("uint8"), pose6D_real)

    plt.imshow(image_bounding_boxes)
    plt.show()

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

    print("Real pose: " + str(pose6D_real))
    print("Predicted pose: " + str(pose6D_predicted))

    print("Real points: " + str(real_bounding_box_points))
    print("Predicted points: " + str(predicted_bounding_box_points))

    real_bounding_box_points_3d = np.asarray([quarternion_to_rotation_matrix(pose6D_real.quaternion)@bb_point + np.squeeze(pose6D_real.translation) for bb_point in bounding_box_points_3d[0]])
    predicted_bounding_box_points_3d = np.asarray([quarternion_to_rotation_matrix(pose6D_predicted.quaternion)@bb_point + np.squeeze(pose6D_predicted.translation) for bb_point in bounding_box_points_3d[0]])

    iou_value = evaluateIoU(real_bounding_box_points_3d, predicted_bounding_box_points_3d,
                            pose6D_real, pose6D_predicted, np.asarray([0.1631425/2., 0.121925/2., 0.17933717/2]), num_sampled_points=1000)
    print("IoU: " + str(iou_value))

    add_value = evaluateADD(real_bounding_box_points_3d, predicted_bounding_box_points_3d)
    print("ADD: " + str(add_value))

    mssd_value = evaluateMSSD(real_bounding_box_points_3d, predicted_bounding_box_points_3d)
    print("MSSD: " + str(mssd_value))

    mspd_value = evaluateMSPD(real_bounding_box_points, predicted_bounding_box_points, renderer.viewport_size)
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


if __name__ == "__main__":
    model = load_model(args.model_path, custom_objects={'loss_color': loss_color, 'loss_error': loss_error})
    print(image_size)
    renderer = SingleView(filepath=args.obj_path, viewport_size=image_size,
                          y_fov=args.y_fov, distance=args.depth, light_bounds=args.light, top_only=bool(args.top_only),
                          roll=args.roll, shift=args.shift)

    make_single_prediction(model, renderer, plot=True)