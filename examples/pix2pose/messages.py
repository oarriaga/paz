from paz.abstract.messages import Box2D, Pose6D


class ObjectHypothesis(object):
    # TODO: Check if class_name, score is the same
    def __init__(self, score=None, class_name=None, box2D=None, pose6D=None):
        self.score = score
        self.class_name = class_name
        self.box2D = box2D
        self.pose6D = pose6D

    @property
    def box2D(self):
        return self._box2D

    @box2D.setter
    def box2D(self, value):
        if not isinstance(value, Box2D):
            raise ValueError('Value must be a Box2D class')

        if self.score is None:
            if value.score is not None:
                self.score = value.score
        else:
            if self.score != value.score:
                raise ValueError('Mismatch score between Hypothesis and Box2D')


        if self.score is None and (value.score is not None):
            self.score = value.score
        elif (self.score is not None) and (value.score is not None):
            if self.score != value.score:
                raise ValueError('Mismatch score between Hypothesis and Box2D')
        if self.class_name is None and (value.class_name is not None):
            self.class_name = value.class_name
        self._box2D = value

    @property
    def pose6D(self):
        return self._pose6D

    @pose6D.setter
    def pose6D(self, value):
        if not isinstance(value, Pose6D):
            raise ValueError('Value must be a Pose6D class')
        if (self.score is None) and (value.score is not None):
            self.score = value.score
        if self.class_name is None and (value.class_name is not None):
            self.class_name = value.class_name
        self._pose6D = value
