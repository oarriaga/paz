import numpy as np
from paz import processors as pr


class GetMultiStageOutputs(pr.Processor):
    """Get Heatmaps and Tags from the model output.
    # Arguments
        model: Model weights trained on HigherHRNet model.
        flipped_joint_order: List of length 17 (number of joints).
            Flipped list of joint order.
        data_with_center: Boolean. True is the model is trained using the
            center.
        image: Numpy array. Input image of shape (H, W)

    # Returns
        heatmaps: Numpy array of shape (1, num_joints, H, W)
        Tags: Numpy array of shape (1, num_joints, H, W)
    """
    def __init__(self, model, flipped_joint_order, with_flip, data_with_center,
                 project2image=True, axes=[0, 3, 1, 2]):
        super(GetMultiStageOutputs, self).__init__()
        self.with_flip = with_flip
        self.predict = pr.SequentialProcessor(
            [pr.Predict(model), pr.TransposeOutput(axes), pr.ScaleOutput(2)])
        self.get_heatmaps = pr.GetHeatmaps(flipped_joint_order)
        self.get_tags = pr.GetTags(flipped_joint_order)
        self.postprocess = pr.SequentialProcessor()
        if data_with_center:
            self.postprocess.add(pr.RemoveLastElement())
        if project2image:
            self.postprocess.add(pr.ScaleOutput(2, full_scaling=True))

    def call(self, image):
        outputs = self.predict(image)
        heatmaps = self.get_heatmaps(outputs, with_flip=False)
        tags = self.get_tags(outputs, with_flip=False)
        if self.with_flip:
            outputs = self.predict(np.flip(image, [2]))
            heatmaps_flip = self.get_heatmaps(outputs, self.with_flip)
            tags_flip = self.get_tags(outputs, self.with_flip)
            heatmaps = [heatmaps, heatmaps_flip]
            tags = [tags, tags_flip]
        heatmaps = self.postprocess(heatmaps)
        tags = self.postprocess(tags)
        return heatmaps, tags


class GetJoints(pr.Processor):
    """Extract out the top k heatmaps and group the joints with their
       respective tags value. Adjust and refine the joint locations by
       removing the margins.
    # Arguments
        max_num_people: Int. Maximum number of person to be detected.
        joint_order: List of length 17 (number of joints).
        heatmaps: Numpy array of shape (1, num_joints, H, W)
        Tags: Numpy array of shape (1, num_joints, H, W, 2)

    # Returns
        grouped_joints: numpy array. joints grouped by tag
        scores: int: score for the joint
    """
    def __init__(self, max_num_people, joint_order, detection_thresh=0.2,
                 tag_thresh=1):
        super(GetJoints, self).__init__()
        self.group_joints = pr.SequentialProcessor(
            [pr.TopKDetections(max_num_people), pr.GroupJointsByTag(
                max_num_people, joint_order, tag_thresh, detection_thresh)])
        self.adjust_joints = pr.AdjustJointsLocations()
        self.get_scores = pr.GetScores()
        self.refine_joints = pr.RefineJointsLocations()

    def call(self, heatmaps, tags, adjust=True, refine=True):
        grouped_joints = self.group_joints(heatmaps, tags)
        if adjust:
            grouped_joints = self.adjust_joints(heatmaps, grouped_joints)[0]
        scores = self.get_scores(grouped_joints)
        if refine:
            grouped_joints = self.refine_joints(
                heatmaps[0], tags[0], grouped_joints)
        return grouped_joints, scores


class InverseTransformJoints(pr.Processor):
    """Inverse the affine transform to get the joint location with respect to
       the input image.
    # Arguments
        grouped_joints: Numpy array. joints grouped by tag
        center: Tuple. center of the imput image
        scale: Float. scaled imput image dimension
        heatmaps: Numpy array of shape (1, num_joints, H, W)

    # Returns
        transformed_joints: joint location with respect to the input image
    """
    def __init__(self):
        super(InverseTransformJoints, self).__init__()
        self.get_affine_transform = pr.GetAffineTransform(inverse=True)
        self.transform_joints = pr.TransformJoints()

    def call(self, grouped_joints, center, scale, shape):
        transform = self.get_affine_transform(center, scale, shape)
        transformed_joints = self.transform_joints(grouped_joints, transform)
        return transformed_joints
