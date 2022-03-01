import numpy as np
from paz import processors as pr
import processors as pe
from paz.models.pose_estimation import HigherHRNet


class DetectHumanPose2D(pr.Processor):
    """Detectect human jonts in a image and draw a skeleton over the image.
    # Arguments
        model: Modle weights trained on HigherHRNet model.
        joint_order: List of length 17 (number of joints).
            where the joints are listed order wise.
        flipped_joint_order: List of length 17 (number of joints).
            Flipped list of joint order.
        dataset: String. Name of the dataset used for training the model.
        data_with_center: Boolean. True is the model is trained using the
            center.
        image: Numpy array. Input image

    # Returns
        dictonary with the following keys:
            image: contains the image with skeleton drawn on it.
            joints: location of joints
            score: score of detection
    """
    def __init__(self, joint_order, flipped_joint_order, dataset,
                 data_with_center, max_num_people=30, with_flip=True,
                 draw=True):
        super(DetectHumanPose2D, self).__init__()
        self.with_flip = with_flip
        self.draw = draw
        self.model = HigherHRNet(weights='COCO')
        self.transform_image = TransformImage()
        self.predict_multi_stage_output = pr.SequentialProcessor(
            [GetMultiStageOutputs(self.model, flipped_joint_order, with_flip,
             data_with_center), AggregateResults(with_flip=self.with_flip)])
        self.heatmaps_parser = HeatmapsParser(max_num_people, joint_order)
        self.transform_joints = InverseTransformJoints()
        self.draw_skeleton = pe.DrawSkeleton(dataset)
        self.extract_joints = pe.ExtractJoints()
        self.wrap = pr.WrapOutput(['image', 'joints', 'scores'])

    def call(self, image):
        resized_image, center, scale = self.transform_image(image)
        heatmaps, tags = self.predict_multi_stage_output(resized_image)
        grouped_joints, scores = self.heatmaps_parser(heatmaps, tags)
        joints = self.transform_joints(grouped_joints, center, scale, heatmaps)
        if self.draw:
            image = self.draw_skeleton(image, joints)
        joints = self.extract_joints(joints)
        return self.wrap(image, joints, scores)


class TransformImage(pr.Processor):
    """Transform the image according to the model input requirement.
    # Arguments
        scaling_factor: Int. scale factor for image dimensions.
        input_size: Int. resize the first dimension of image to input size.
        inverse: Boolean. Reverse the affine transform input.
        image: Numpy array. Input image

    # Returns
        image: resized and transformed image
        center: center of the image
        scale: scaled image dimensions
    """
    def __init__(self, scaling_factor=200, input_size=512, inverse=False):
        super(TransformImage, self).__init__()
        self.get_image_center = pe.GetImageCenter()
        self.get_size = pe.GetTransformationSize(input_size)
        self.get_scale = pe.GetTransformationScale(scaling_factor)
        self.get_affine_transform = pe.GetAffineTransform(inverse)
        self.transform_image = pr.SequentialProcessor(
            [pe.WarpAffine(), pe.ImagenetPreprocessInput(), pr.ExpandDims(0)])

    def call(self, image):
        center = self.get_image_center(image)
        size = self.get_size(image)
        scale = self.get_scale(image, size)
        transform = self.get_affine_transform(center, scale, size)
        image = self.transform_image(image, transform, size)
        return image, center, scale


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
            [pr.Predict(model), pe.TransposeOutput(axes), pe.ScaleOutput(2)])
        self.get_heatmaps = pe.GetHeatmaps(flipped_joint_order)
        self.get_tags = pe.GetTags(flipped_joint_order)
        self.postprocess = pr.SequentialProcessor()
        if data_with_center:
            self.postprocess.add(pe.RemoveLastElement())
        if project2image:
            self.postprocess.add(pe.ScaleInput(2))

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


class AggregateResults(pr.Processor):
    """Aggregate heatmaps and tags to get final heatmaps and tags for
       processing.
    # Arguments
        heatmaps: Numpy array of shape (1, num_joints, H, W)
        Tags: Numpy array of shape (1, num_joints, H, W)

    # Returns
        heatmaps: Numpy array of shape (1, num_joints, H, W)
        Tags: Numpy array of shape (1, num_joints, H, W, 2)
    """

    def __init__(self, project2image=True, with_flip=False):
        super(AggregateResults, self).__init__()
        self.aggregate_heatmaps = pr.SequentialProcessor(
            [pe.CalculateHeatmapsAverage(with_flip),
             pe.AggregateHeatmapsAverage(project2image)])
        self.aggregate_tags = pr.SequentialProcessor(
            [pe.ExpandTagsDimension(), pr.Concatenate(4)])

    def call(self, heatmaps, tags):
        return self.aggregate_heatmaps(heatmaps), self.aggregate_tags(tags)


class HeatmapsParser(pr.Processor):
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
        super(HeatmapsParser, self).__init__()
        self.group_joints = pr.SequentialProcessor(
            [pe.TopKDetections(max_num_people),
             pe.GroupJointsByTag(max_num_people, joint_order, tag_thresh,
                                 detection_thresh)])
        self.adjust_joints = pe.AdjustJointsLocations()
        self.get_scores = pe.GetScores()
        self.refine_joints = pe.RefineJointsLocations()

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
        self.get_affine_transform = pe.GetAffineTransform(inverse=True)
        self.transform_joints = pe.TransformJoints()

    def call(self, grouped_joints, center, scale, heatmaps):
        heatmaps_size = [heatmaps.shape[3], heatmaps.shape[2]]
        transform = self.get_affine_transform(center, scale, heatmaps_size)
        transformed_joints = self.transform_joints(grouped_joints, transform)
        return transformed_joints
