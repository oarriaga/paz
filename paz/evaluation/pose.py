import os
import numpy as np
from tensorflow.keras.callbacks import Callback
from ..backend.image import load_image
from ..backend.groups import quaternion_to_rotation_matrix


class EvaluateADD(Callback):
    """Callback for evaluating the pose error on ADD/ADI metric.

    # Arguments
        experiment_path: String. Path in which the images will be saved.
        evaluation_data_manager: Object of type dataset loader
            e.g. Linemod.
        pipeline: Function that takes as input an element of ''images''
            and outputs a ''Dict'' with inferences.
        mesh_points: nx3 ndarray with 3D model points.
        object_diameter: Float, diameter of the object.
        evaluation_period: Int, interval for pose error
            metric calculation.
        topic: Key to the ''inferences'' dictionary containing as value
            the drawn inferences.
        verbose: Integer. If is bigger than 1 messages
            would be displayed.
    """
    def __init__(self, experiment_path, evaluation_data_manager, pipeline,
                 mesh_points, object_diameter, evaluation_period,
                 topic='poses6D', verbose=1):
        self.experiment_path = experiment_path
        self.evaluation_data_manager = evaluation_data_manager
        self.images = load_test_images(evaluation_data_manager)
        self.gt_poses = load_gt_poses(evaluation_data_manager)
        self.pipeline = pipeline
        self.mesh_points = mesh_points
        self.object_diameter = object_diameter
        self.evaluation_period = evaluation_period
        self.topic = topic
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.evaluation_period == 0:
            ADD_error_sum, ADD_accuracy_sum, valid_predictions = (0.0, 0.0, 0)
            for image, gt_pose in zip(self.images, self.gt_poses):
                inferences = self.pipeline(image.copy())
                pose6D = inferences[self.topic]
                if pose6D:
                    ADD_error, is_correct = compute_ADD_metric(
                        gt_pose, pose6D[0], self.mesh_points,
                        self.object_diameter)
                    ADD_error_sum = ADD_error_sum + ADD_error
                    ADD_accuracy_sum = ADD_accuracy_sum + float(is_correct)
                    valid_predictions = valid_predictions + 1

            error_path = os.path.join(self.experiment_path, 'ADD_error.txt')
            if valid_predictions > 0:
                average_ADD_error = ADD_error_sum / valid_predictions
                average_accuracy = ADD_accuracy_sum / len(self.gt_poses)
                with open(error_path, 'a') as filer:
                    filer.write('epoch: {}'.format(epoch))
                    filer.write('\nEstimated ADD error: {}'.format(
                        average_ADD_error))
                    filer.write('\nEstimated ADD accuracy: {}\n\n'.format(
                        average_accuracy))
            else:
                average_ADD_error = None
                average_accuracy = None
            if self.verbose:
                print('Estimated ADD error: {}'.format(average_ADD_error))
                print('Estimated ADD accuracy: {}'.format(average_accuracy))


class EvaluateADI(Callback):
    """Callback for evaluating the pose error on ADD/ADI metric.

    # Arguments
        experiment_path: String. Path in which the images will be saved.
        evaluation_data_manager: Object of type dataset loader
            e.g. Linemod.
        pipeline: Function that takes as input an element of ''images''
            and outputs a ''Dict'' with inferences.
        mesh_points: nx3 ndarray with 3D model points.
        object_diameter: Float, diameter of the object.
        evaluation_period: Int, interval for pose error
            metric calculation.
        topic: Key to the ''inferences'' dictionary containing as value
            the drawn inferences.
        verbose: Integer. If is bigger than 1 messages
            would be displayed.
    """
    def __init__(self, experiment_path, evaluation_data_manager, pipeline,
                 mesh_points, object_diameter, evaluation_period,
                 topic='poses6D', verbose=1):
        self.experiment_path = experiment_path
        self.evaluation_data_manager = evaluation_data_manager
        self.images = load_test_images(evaluation_data_manager)
        self.gt_poses = load_gt_poses(evaluation_data_manager)
        self.pipeline = pipeline
        self.mesh_points = mesh_points
        self.object_diameter = object_diameter
        self.evaluation_period = evaluation_period
        self.topic = topic
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.evaluation_period == 0:
            ADI_error_sum, valid_predictions = (0.0, 0)
            for image, gt_pose in zip(self.images, self.gt_poses):
                inferences = self.pipeline(image.copy())
                pose6D = inferences[self.topic]
                if pose6D:
                    error = compute_ADI_metric(
                        gt_pose, pose6D[0], self.mesh_points,
                        self.object_diameter)
                    ADI_error_sum = ADI_error_sum + error
                    valid_predictions = valid_predictions + 1

            error_path = os.path.join(self.experiment_path, 'ADI_error.txt')
            if valid_predictions > 0:
                average_error = ADI_error_sum / valid_predictions
                with open(error_path, 'a') as filer:
                    filer.write('epoch: {}'.format(epoch))
                    filer.write('\nEstimated ADI error: {}'.format(
                        average_error))
            else:
                average_error = None
            if self.verbose:
                print('Estimated {} error: {}'.format(
                    self.metric, average_error))


def load_test_images(data_manager):
    """Loads test images from a given evaluation data manager.

    # Arguments
        data_manager: Object of type dataset loader e.g. Linemod.

    # Returns
        List, containinig images for metric evaluation.
    """
    data = data_manager.load_data()
    images = [load_image(datum['image']) for datum in data]
    return images


def load_gt_poses(data_manager):
    """Loads ground truth poses from a given evaluation data manager.

    # Arguments
        data_manager: Object of type dataset loader e.g. Linemod.

    # Returns
        List, containinig ground truth poses for metric evaluation.
    """
    data = data_manager.load_data()
    gt_poses = []
    for datum in data:
        rotation = datum['rotation']
        rotation_matrix = rotation.reshape((3, 3))
        translation = datum['translation_raw']
        gt_pose = np.concatenate((rotation_matrix, translation.T), axis=1)
        gt_poses.append(gt_pose)
    return gt_poses


def transform_mesh_points(mesh_points, rotation, translation):
    """Transforms object points.

    # Arguments
        mesh_points: Array of shape `(n, 3)` with 3D model points.
        rotation: Array of shape `(3, 3)` rotation matrix.
        translation: Array of shape `(1, 3)` translation vector.

    # Returns
        Array of shape `(n, 3)` with transformed 3D model points.
    """
    assert (mesh_points.shape[1] == 3)
    pts_t = rotation.dot(mesh_points.T) + translation.reshape((3, 1))
    return pts_t.T


def compute_ADD_metric(gt_pose, pose6D, mesh_points, object_diameter):
    ADD_error = compute_ADD_error(gt_pose, pose6D, mesh_points)
    is_correct = check_ADD(ADD_error, object_diameter)
    return [ADD_error, is_correct]


def compute_ADD_error(pose_true, pose_pred, mesh_points):
    """Calculates ADD error.

    # Arguments
        true_pose: Pose6D real pose.
        pred_pose: Pose6D, predicted pose.
        mesh_points: Array of shape `(n, 3)` with 3D model points.

    # Returns
        Array of shape `()` with ADD error.
    """
    quaternion = pose_pred.quaternion
    translation_pred = pose_pred.translation
    rotation_pred = quaternion_to_rotation_matrix(quaternion)
    mesh_pred = transform_mesh_points(mesh_points, rotation_pred,
                                      translation_pred)
    rotation_true = pose_true[:3, :3]
    translation_true = pose_true[:3, 3]
    mesh_true = transform_mesh_points(mesh_points, rotation_true,
                                      translation_true)

    return np.linalg.norm(mesh_pred - mesh_true, axis=1).mean()


def check_ADD(ADD_error, diameter, diameter_threshold=0.1):
    """Checks if ADD error is within tolerance. Returns `True`
    if ADD error is within tolerance else `False`.

    # Arguments
        ADD_error: Float, the ADD error.
        diameter: Float, diameter of the object.
        diameter_threshold: Float, diameter tolerance in %.

    # Returns
        is_correct: Bool.
    """
    if ADD_error <= (diameter * diameter_threshold):
        is_correct = True
    else:
        is_correct = False
    return is_correct


def compute_ADI_metric(pose_true, pose_pred, mesh_points, *args):
    """Calculate The ADI error. Calculate distances to the
    nearest neighbors from vertices in the ground-truth pose to
    vertices in the estimated pose.

    # Arguments
        true_pose: Pose6D real pose.
        pred_pose: Pose6D, predicted pose.
        mesh_points: Array of shape `(n, 3)` with 3D model points.

    # Returns
        Array of shape `()` with ADI error.
    """
    quaternion = pose_pred.quaternion
    translation_pred = pose_pred.translation
    rotation_pred = quaternion_to_rotation_matrix(quaternion)
    mesh_pred = transform_mesh_points(mesh_points, rotation_pred,
                                      translation_pred)
    rotation_true = pose_true[:3, :3]
    translation_true = pose_true[:3, 3]
    mesh_true = transform_mesh_points(mesh_points, rotation_true,
                                      translation_true)
    return compute_nearest_distance(mesh_pred, mesh_true, k=1)


def compute_nearest_distance(X, Y, k=1):
    """Calculates `k` nearest neighbour distances of points `X`
    with that of `Y`.

    # Arguments
        X: Array of shape `(n, 3)`.
        Y: Array of shape `(n, 3)`.
        k: Int, number of neighbours to consider.

    # Returns
        Array of shape `()` mean distances.
    """
    distance_matrix = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=-1)
    distance_matrix_sorted = np.sort(distance_matrix, axis=-1)
    top_k_distances = distance_matrix_sorted[:, :k]
    return np.mean(top_k_distances)
