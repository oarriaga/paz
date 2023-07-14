import os
import numpy as np
from scipy import spatial
from tensorflow.keras.callbacks import Callback


def transform_mesh_points(mesh_points, rotation, translation):
    """Transforms the object points
      # Arguments
          mesh_points: nx3 ndarray with 3D model points.
          rotaion: Rotation matrix
          translation: Translation vector

      # Returns
          Transformed model
      """
    assert (mesh_points.shape[1] == 3)
    pts_t = rotation.dot(mesh_points.T) + translation.reshape((3, 1))
    return pts_t.T


def compute_ADD(true_pose, pred_pose, mesh_points):
    """Calculate The ADD error.
      # Arguments
          true_pose: Real pose
          pred_pose: Predicted pose
          mesh_pts: nx3 ndarray with 3D model points.
      # Returns
          Return ADD error
    """
    pred_rotation = pred_pose[:3, :3]
    pred_translation = pred_pose[:3, 3]
    pred_mesh = transform_mesh_points(mesh_points, pred_rotation,
                                      pred_translation)

    true_rotation = true_pose[:3, :3]
    true_translation = true_pose[:3, 3]
    true_mesh = transform_mesh_points(mesh_points, true_rotation,
                                      true_translation)

    error = np.linalg.norm(pred_mesh - true_mesh, axis=1).mean()
    return error


def compute_ADI(true_pose, pred_pose, mesh_points):
    """Calculate The ADI error.
       Calculate distances to the nearest neighbors from vertices in the
       ground-truth pose to vertices in the estimated pose.
      # Arguments
          true_pose: Real pose
          pred_pose: Predicted pose
          mesh_pts: nx3 ndarray with 3D model points.
      # Returns
          Return ADI error
      """
    pred_rotation = pred_pose[:3, :3]
    pred_translation = pred_pose[:3, 3]
    pred_mesh = transform_mesh_points(mesh_points, pred_rotation,
                                      pred_translation)

    true_rotation = true_pose[:3, :3]
    true_translation = true_pose[:3, 3]
    true_mesh = transform_mesh_points(mesh_points, true_rotation,
                                      true_translation)
    nn_index = spatial.cKDTree(pred_mesh)
    nn_dists, _ = nn_index.query(true_mesh, k=1)

    error = nn_dists.mean()
    return error


class EvaluatePoseError(Callback):
    """Callback for evaluating the pose error on ADD and ADI metric.

    # Arguments
        experiment_path: String. Path in which the images will be saved.
        images: List of numpy arrays of shape.
        pipeline: Function that takes as input an element of ''images''
            and outputs a ''Dict'' with inferences.
        mesh_points: nx3 ndarray with 3D model points.
        topic: Key to the ''inferences'' dictionary containing as value the
            drawn inferences.
        verbose: Integer. If is bigger than 1 messages would be displayed.
    """
    def __init__(self, experiment_path, images, pipeline, mesh_points,
                 topic='pose6D', verbose=1):
        super(EvaluatePoseError, self).__init__()
        self.experiment_path = experiment_path
        self.images = images
        self.pipeline = pipeline
        self.mesh_points = mesh_points
        self.topic = topic
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        for image_arg, image in enumerate(self.images):
            inferences = self.pipeline(image.copy())
            pose6D = inferences[self.topic]
            gt_file = 'gt_pose_%03d.npy' % image_arg
            image_directory = os.path.join(self.experiment_path,
                                           'original_images')
            gt_pose = np.load(os.path.join(image_directory, gt_file))
            add_error = compute_ADD(gt_pose, pose6D, self.mesh_points)
            adi_error = compute_ADI(gt_pose, pose6D, self.mesh_points)

            with open(os.path.join(self.experiment_path,
                                   'error.txt'), 'w') as filer:
                filer.write('epoch: %d\n' % epoch)
                filer.write('Estimated ADD error: %f\n' % add_error)
                filer.write('Estimated ADI error: %f\n\n' % adi_error)
        if self.verbose:
            print('Estimated ADD error:', add_error)
            print('Estimated ADI error:', adi_error)
