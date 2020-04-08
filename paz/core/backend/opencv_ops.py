import colorsys
import random
import cv2
import os

GREEN = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE = cv2.LINE_AA
BGR2RGB = cv2.COLOR_BGR2RGB
RGB2BGR = cv2.COLOR_RGB2BGR
BGR2HSV = cv2.COLOR_BGR2HSV
RGB2HSV = cv2.COLOR_RGB2HSV
HSV2RGB = cv2.COLOR_HSV2RGB
HSV2BGR = cv2.COLOR_HSV2BGR
BGR2GRAY = cv2.COLOR_BGR2GRAY
IMREAD_COLOR = cv2.IMREAD_COLOR
UPNP = cv2.SOLVEPNP_UPNP


def cascade_classifier(path):
    """Cascade classifier with detectMultiScale() method for inference.
    # Arguments
        path: String. Path to default openCV XML format.
    """
    return cv2.CascadeClassifier(path)


def load_image(filepath, flags=cv2.IMREAD_COLOR):
    """Loads an image.
    # Arguments
        filepath: string with image path
        flags: Integers indicating flags about how to read image:
            1 or cv2.IMREAD_COLOR for BGR image.
            0 or cv2.IMREAD_GRAYSCALE for grayscale image.
           -1 or cv2.IMREAD_UNCHANGED for BGR with alpha-channel.
    # Returns
        Image as numpy array.
    """
    return cv2.imread(filepath, flags)


def resize_image(image, shape):
    """ Resizes image.
    # Arguments
        image: Numpy array.
        shape: List of two integer elements indicating new shape.
    """
    return cv2.resize(image, shape)


def save_image(filepath, image, *args):
    """Saves an image.
    # Arguments
        filepath: String with image path. It should include postfix e.g. .png
        image: Numpy array.
    """
    return cv2.imwrite(filepath, image, *args)


def save_images(save_path, images):
    """Saves multiple images in a directory
    # Arguments
        save_path: String. Path to directory. If path does not exist it will
        be created.
        images: List of numpy arrays.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for arg, image in enumerate(images):
        save_image(os.path.join(save_path, 'image_%03d.png' % arg), image)


def convert_image(image, flag):
    """Converts image to a different color space
    # Arguments
        image: Numpy array
        flag: OpenCV color flag e.g. cv2.COLOR_BGR2RGB or BGR2RGB
    """
    return cv2.cvtColor(image, flag)


def show_image(image, name='image', wait=True):
    """ Shows image in an external window.
    # Arguments
        image: Numpy array
        name: String indicating the window name.
    """
    cv2.imshow(name, image)
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def warp_affine(image, matrix, fill_color=[0, 0, 0]):
    """ Transforms `image` using an affine `matrix` transformation.
    # Arguments
        image: Numpy array.
        matrix: Numpy array of shape (2,3) indicating affine transformation.
        fill_color: List/tuple representing BGR use for filling empty space.
    """
    height, width = image.shape[:2]
    return cv2.warpAffine(
        image, matrix, (width, height), borderValue=fill_color)


def solve_PNP(points3D, points2D, camera, solver):
    """Calculates 6D pose from 3D points and 2D keypoints correspondences.
    # Arguments
        points: Numpy array of shape (num_points, 3).
            Model 3D points known in advance.
        keypoints: Numpy array of shape (num_points, 2).
            Predicted 2D keypoints of object
        camera intrinsics: Numpy array of shape (3, 3) calculated from
        the openCV calibrateCamera function
        solver: Flag from e.g openCV.SOLVEPNP_UPNP
        distortion: Numpy array of shape of 5 elements calculated from
        the openCV calibrateCamera function

    # Returns
        A list containing success flag, rotation and translation components
        of the 6D pose.

    # References
        https://docs.opencv.org/2.4/modules/calib3d/doc/calib3d.html
    """
    return cv2.solvePnP(points3D, points2D, camera.intrinsics,
                        camera.distortion, None, None, False, solver)


def project_points3D(points3D, pose6D, camera):
    point2D, jacobian = cv2.projectPoints(
        points3D, pose6D.rotation_vector, pose6D.translation,
        camera.intrinsics, camera.distortion)
    return point2D
