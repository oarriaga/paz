import os
import cv2
from paz.backend.camera import Camera
from paz.backend.image import load_image, show_image
from scipy.optimize import least_squares
from tensorflow.keras.utils import get_file
from paz.pipelines import EstimateHumanPose
from paz.processors import OptimizeHumanPose3D
from paz.datasets.human36m import args_to_joints3D
from viz import visualize
import numpy as np
import paz.processors as pr
from paz.backend.keypoints import project_to_image
from paz.backend.image import draw_line


def pose3D_to_pose6D(poses3D):
    # extract hip locations
    right_hip = poses3D[1]
    left_hip = poses3D[6]
    thorax = poses3D[13]

    # Find human orientation by placing a coordinate system
    # along hip joints
    v1 = right_hip - left_hip
    projection_v = thorax - left_hip
    a = np.dot(v1, projection_v) / np.linalg.norm(v1) ** 2
    projected_point = left_hip + a * v1
    v2 = thorax - projected_point

    # make unit vectors of v1 and v2
    v1_hat = v1 / np.linalg.norm(v1)
    v2_hat = v2 / np.linalg.norm(v2)
    # v2 x v1
    v3 = np.cross(v2_hat, v1_hat)

    # Rotation matrix
    R = np.column_stack((v1_hat, v3, v2_hat))  # X,Y,Z
    # quaternions = pr.quaternion_from_matrix(R)
    translation = (poses3D[0]/1e3).tolist()
    return R, translation


def draw_human_pose6D(R, translation, image, camaera_intrinsics):
    points3D = np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]])
    points2D = project_to_image(R, translation, points3D, camaera_intrinsics)
    points2D = points2D.astype(np.int)

    x = points2D[0]
    y = points2D[1]
    z = points2D[2]

    x_hat = (x / np.linalg.norm(x) * 60).astype(np.int)
    y_hat = (y / np.linalg.norm(y) * 60).astype(np.int)
    z_hat = (z / np.linalg.norm(z) * 60).astype(np.int)

    offset = [50, 50]
    image = draw_line(image, offset, x_hat + offset,
                      color=[255, 0, 0], thickness=4)
    image = draw_line(image, offset, y_hat + offset,
                      color=[0, 255, 0], thickness=4)
    image = draw_line(image, offset, z_hat + offset,
                      color=[0, 0, 255], thickness=4)
    return image


class TestHumanPose(pr.Processor):
    def __init__(self, camera_intrinsics):
        super(TestHumanPose, self).__init__()
        self.estimatate_pose = EstimateHumanPose()
        self.optimize = OptimizeHumanPose3D(args_to_joints3D,
                                            least_squares, camera_intrinsics)
        self.draw_text = pr.DrawText(scale=0.5, thickness=1)
        self.wrap = pr.WrapOutput(['image'])

    def call(self, image):
        keypoints = self.estimatate_pose(image)
        keypoints2D = keypoints['keypoints2D']
        keypoints3D = keypoints['keypoints3D']
        image = keypoints['image2D']
        keypoints2D, joints3D, optimized_poses3D = self.optimize(keypoints3D,
                                                                 keypoints2D)
        print('optimized pose')
        print(optimized_poses3D)
        R, translation = pose3D_to_pose6D(keypoints3D[0])
        formatted_translation = ["%.2f" % item for item in translation]
        image = self.draw_text(image, str(formatted_translation), (30, 30))
        image = draw_human_pose6D(R, translation, image, camera.intrinsics)
        return self.wrap(image)


# URL = ('https://github.com/oarriaga/altamira-data/releases/download'
#        '/v0.17/multiple_persons_posing.png')

# filename = os.path.basename(URL)
# fullpath = get_file(filename, URL, cache_subdir='paz/tests')
# image = load_image(fullpath)

image = load_image("../human_pose_estimation_3D/person_standing.jpg")
# image = cv2.resize(image, (480, 720))
H, W = image.shape[:2]
camera = Camera()
camera.intrinsics_from_HFOV(HFOV=70, image_shape=[600, 800])

pipeline = TestHumanPose(camera.intrinsics)
inference = pipeline(image)
show_image(inference['image'])
