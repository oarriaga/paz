import numpy as np
from paz.abstract import Processor
from paz.backend.keypoints import project_points3D
from paz.backend.keypoints import build_cube_points3D
from paz.backend.image import draw_cube
from paz.processors import DrawBoxes3D


class DrawBoxes3D(Processor):
    def __init__(self, camera, class_to_dimensions, thickness=1):
        """Draw boxes 3D of multiple objects

        # Arguments
            camera: Instance of ``paz.backend.camera.Camera''.
            class_to_dimensions: Dictionary that has as keys the
                class names and as value a list [model_height, model_width]
            thickness: Int. Thickness of 3D box
        """
        super(DrawBoxes3D, self).__init__()
        self.camera = camera
        self.class_to_dimensions = class_to_dimensions
        self.class_to_points = self._build_points(self.class_to_dimensions)
        self.thickness = thickness

    def _build_points(self, class_to_dimensions):
        class_to_cube3D = {}
        print(class_to_dimensions)
        for class_name, dimensions in class_to_dimensions.items():
            width, height, depth = dimensions
            cube_points3D = build_cube_points3D(width, height, depth)
            class_to_cube3D[class_name] = cube_points3D
        return class_to_cube3D

    def call(self, image, pose6D):
        points3D = self.class_to_points[pose6D.class_name]
        points2D = project_points3D(points3D, pose6D, self.camera)
        points2D = points2D.astype(np.int32)
        draw_cube(image, points2D, thickness=self.thickness)
        return image
