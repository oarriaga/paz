from ..backend.groups.quaternion import rotation_vector_to_quaternion


class Box2D(object):
    """Bounding box 2D coordinates with class label and score.

    # Properties
        coordinates: List of float/integers indicating the
            ``[x_min, y_min, x_max, y_max]`` coordinates.
        score: Float. Indicates the score of label associated to the box.
        class_name: String indicating the class label name of the object.

    # Methods
        contains()
    """
    def __init__(self, coordinates, score, class_name=None):
        x_min, y_min, x_max, y_max = coordinates
        self.coordinates = coordinates
        self.class_name = class_name
        self.score = score

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates):
        x_min, y_min, x_max, y_max = coordinates
        if x_min >= x_max:
            raise ValueError('Invalid coordinate input x_min >= x_max')
        if y_min >= y_max:
            raise ValueError('Invalid coordinate input y_min >= y_max')

        self._coordinates = coordinates

    @property
    def class_name(self):
        return self._class_name

    @class_name.setter
    def class_name(self, class_name):
        self._class_name = class_name

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, score):
        self._score = score

    @property
    def center(self):
        x_center = (self._coordinates[0] + self._coordinates[2]) / 2.0
        y_center = (self._coordinates[1] + self._coordinates[3]) / 2.0
        return x_center, y_center

    @property
    def width(self):
        return abs(self.coordinates[2] - self.coordinates[0])

    @property
    def height(self):
        return abs(self.coordinates[3] - self.coordinates[1])

    def __repr__(self):
        return "Box2D({}, {}, {}, {}, {}, {})".format(
            self.coordinates[0], self.coordinates[1],
            self.coordinates[2], self.coordinates[3],
            self.score, self.class_name)

    def contains(self, point):
        """Checks if point is inside bounding box.

        # Arguments
            point: Numpy array of size 2.

        # Returns
            Boolean. 'True' if 'point' is inside bounding box.
                'False' otherwise.
        """
        assert len(point) == 2
        x_min, y_min, x_max, y_max = self.coordinates
        inside_range_x = (point[0] >= x_min) and (point[0] <= x_max)
        inside_range_y = (point[1] >= y_min) and (point[1] <= y_max)
        return (inside_range_x and inside_range_y)


class Pose6D(object):
    """ Pose estimation results with 6D coordinates.

        # Properties
            quaternion: List of 4 floats indicating (w, x, y, z) components.
            translation: List of 3 floats indicating (x, y, z)
                translation components.
            class_name: String or ``None`` indicating the class label name of
                the object.

        # Class Methods
            from_rotation_vector: Instantiates a ``Pose6D`` object using a
                rotation and a translation vector.
    """
    def __init__(self, quaternion, translation, class_name=None):
        self.quaternion = quaternion
        self.translation = translation
        self.class_name = class_name

    @property
    def quaternion(self):
        return self._quaternion

    @quaternion.setter
    def quaternion(self, coordinates):
        self._quaternion = coordinates

    @property
    def translation(self):
        return self._translation

    @translation.setter
    def translation(self, coordinates):
        self._translation = coordinates

    @property
    def class_name(self):
        return self._class_name

    @class_name.setter
    def class_name(self, class_name):
        self._class_name = class_name

    @classmethod
    def from_rotation_vector(
            cls, rotation_vector, translation, class_name=None):
        quaternion = rotation_vector_to_quaternion(rotation_vector)
        pose6D = cls(quaternion, translation, class_name)
        pose6D.rotation_vector = rotation_vector
        return pose6D

    def __repr__(self):
        quaternion_message = ' Quaternion: ({}, {}, {}, {}) '.format(
            self.quaternion[0], self.quaternion[1],
            self.quaternion[2], self.quaternion[3])
        translation_message = ' Translation: ({}, {}, {}) '.format(
            self.translation[0], self.translation[1], self.translation[2])
        pose_message = ['Pose6D: ', quaternion_message, translation_message]
        pose_message = '\n \t'.join(pose_message)
        return pose_message


class Keypoint3D(object):
    def __init__(self, coordinates, class_name=None):
        coordinates = coordinates
        class_name = class_name

    @property
    def coordinates(self, coordinates):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates):
        num_keypoints = len(coordinates)
        if num_keypoints != 3:
            raise ValueError('Invalid 3D Keypoint length:', num_keypoints)
        self._coordinates = coordinates

    def project():
        raise NotImplementedError

    def unproject():
        raise NotImplementedError
