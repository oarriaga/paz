import numpy as np
from paz import processors as pr
import processors as pe


class DetectHumanPose2D(pr.Processor):
    """Detectect human jonts in a image and draw a skeleton over the image.
    # Arguments
        model: Modle weights trained on HigherHRNet model.
        joint_order: List of length 17 (number of joints).
            where the joints are listed order wise.
        fliped_joint_order: List of length 17 (number of joints).
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
    def __init__(self, model, joint_order, fliped_joint_order, dataset,
                 data_with_center, max_num_people=30, with_flip=True,
                 draw=True):
        super(DetectHumanPose2D, self).__init__()
        self.with_flip = with_flip
        self.draw = draw
        self.preprocess_image = PreprocessImage()
        self.predict_multi_stage_output = pr.SequentialProcessor(
            [GetMultiStageOutputs(model, fliped_joint_order, data_with_center,
             with_flip), AggregateResults(with_flip=self.with_flip)])
        self.heatmaps_parser = HeatmapsParser(max_num_people, joint_order)
        self.transform_joints = TransformJoints()
        self.draw_skeleton = pe.DrawSkeleton(dataset)
        self.wrap = pr.WrapOutput(['image', 'joints', 'scores'])

    def call(self, image):
        resized_image, center, scale = self.preprocess_image(image)
        heatmaps, tags = self.predict_multi_stage_output(resized_image)
        grouped_joints, scores = self.heatmaps_parser(heatmaps, tags)
        joints = self.transform_joints(grouped_joints, center, scale, heatmaps)
        if self.draw:
            image = self.draw_skeleton(image, joints)
        return self.wrap(image, joints, scores)


class PreprocessImage(pr.Processor):
    """Preprocess the image accourding to the model input requirement.
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
        super(PreprocessImage, self).__init__()
        self.get_image_center = pe.GetImageCenter()
        self.resize_dims = pe.ResizeDims(input_size)
        self.scale_dims = pe.ScaleDims(scaling_factor)
        self.get_affine_transform = pe.GetAffineTransform(inverse)
        self.transform_image = pr.SequentialProcessor(
            [pe.WarpAffine(), pe.ImagenetPreprocessInput(), pr.ExpandDims(0)])

    def call(self, image):
        center = self.get_image_center(image)
        size = self.resize_dims(image)
        scale = self.scale_dims(image, size)
        transform = self.get_affine_transform(center, scale, size)
        image = self.transform_image(image, transform, size)
        return image, center, scale


class GetMultiStageOutputs(pr.Processor):
    """Get Heatmaps and Tags from the model output.
    # Arguments
        model: Modle weights trained on HigherHRNet model.
        fliped_joint_order: List of length 17 (number of joints).
            Flipped list of joint order.
        data_with_center: Boolean. True is the model is trained using the
            center.
        image: Numpy array. Input image of shape (H, W)

    # Returns
        heatmaps: Numpy array of shape (1, H/2, W/2, num_joints)
        Tgas: Numpy array of shape (1, H/2, W/2, num_joints)
    """
    def __init__(self, model, fliped_joint_order, data_with_center, with_flip,
                 tag_per_joint=True, project2image=True):
        super(GetMultiStageOutputs, self).__init__()
        self.with_flip = with_flip
        self.heatmaps, self.tags = [], []
        self.predict = pr.SequentialProcessor(
            [pr.Predict(model), pe.ScaleOutput(2)])
        self.get_heatmaps = pr.SequentialProcessor(
            [pe.GetHeatmapsAverage(fliped_joint_order), pe.UpdateHeatmaps()])
        self.get_tags = pe.GetTags(fliped_joint_order, tag_per_joint)
        self.postprocess = pr.SequentialProcessor()
        if data_with_center:
            self.postprocess.add(pe.RemoveLastElement())
        if project2image:
            self.postprocess.add(pe.ScaleImage(2))

    # move with flip to __init__.
    def call(self, image):
        outputs = self.predict(image)
        heatmaps = self.get_heatmaps(outputs, self.heatmaps, with_flip=False)
        tags = self.get_tags(outputs, self.tags, with_flip=False)
        if self.with_flip:
            outputs = self.predict(np.flip(image, [2]))
            heatmaps = self.get_heatmaps(outputs, heatmaps, self.with_flip)
            tags = self.get_tags(outputs, tags, self.with_flip)
        heatmaps = self.postprocess(heatmaps)
        tags = self.postprocess(tags)
        return heatmaps, tags


class AggregateResults(pr.Processor):
    """Aggregate heatmaps and tags to get final heatmaps and tags for
       processing.
    # Arguments
        heatmaps: Numpy array of shape (1, H/2, W/2, num_joints)
        Tgas: Numpy array of shape (1, H/2, W/2, num_joints)

    # Returns
        heatmaps: Numpy array of shape (1, H/2, W/2, num_joints, 1)
        Tgas: Numpy array of shape (1, H/2, W/2, num_joints, 1)
    """

    def __init__(self, project2image=True, with_flip=False):
        super(AggregateResults, self).__init__()
        self.aggregate_tags = pr.SequentialProcessor(
            [pe.ExpandTagsDimension(), pe.Concatenate(4)])
        self.aggregate_heatmaps = pr.SequentialProcessor(
            [pe.CalculateHeatmapsAverage(with_flip),
             pe.AggregateHeatmapsAverage(project2image)])

    def call(self, heatmaps, tags):
        tags = self.aggregate_tags(tags)
        heatmaps = self.aggregate_heatmaps(heatmaps)
        return heatmaps, tags


class HeatmapsParser(pr.Processor):
    """Extract out the top k heatmaps and group the joints with their
       respective tags value. Adjust and refine the joint locations by
       removing the margins.
    # Arguments
        max_num_people: Int. Maximum number of person to be detected.
        joint_order: List of length 17 (number of joints).
        heatmaps: Numpy array of shape (1, H/2, W/2, num_joints)
        Tgas: Numpy array of shape (1, H/2, W/2, num_joints)

    # Returns
        grouped_joints: numpy array. joints grouped by tag
        scores: int: score for the joint
    """
    def __init__(self, max_num_people, joint_order, detection_thresh=0.2,
                 tag_thresh=1, tag_per_joint=True):
        super(HeatmapsParser, self).__init__()
        self.group_joints = pr.SequentialProcessor(
            [pe.TopKDetections(max_num_people, tag_per_joint, joint_order),
             pe.GroupJointsByTag(max_num_people, joint_order, tag_thresh,
                                 detection_thresh)])
        self.adjust_joints = pe.AdjustJointsLocations()
        self.get_scores = pe.GetScores()
        self.tile_array = pe.TileArray(tag_per_joint)
        self.refine_joints = pe.RefineJointsLocations()

    def call(self, heatmaps, tags, adjust=True, refine=True):
        grouped_joints = self.group_joints(heatmaps, tags)
        if adjust:
            grouped_joints = self.adjust_joints(heatmaps, grouped_joints)[0]
        scores = self.get_scores(grouped_joints)
        heatmaps, tags = self.tile_array(heatmaps, tags)
        if refine:
            grouped_joints = self.refine_joints(heatmaps, tags, grouped_joints)
        return grouped_joints, scores


class TransformJoints(pr.Processor):
    """inverse the affine transform to get the joint location with respect to
       the input image.
    # Arguments
        grouped_joints: Numpy array. joints grouped by tag
        center: Tuple. center of the imput image
        scale: Float. scaled imput image dimension
        heatmaps: Numpy array of shape (1, H/2, W/2, num_joints)

    # Returns
        transformed_joints: joint location with respect to the input image
    """
    def __init__(self):
        super(TransformJoints, self).__init__()
        self.transformed_joints = []
        self.get_affine_transform = pe.GetAffineTransform(inverse=True)
        self.transform_point = pe.TransformPoint()

    def call(self, grouped_joints, center, scale, heatmaps):
        heatmaps_size = [heatmaps.shape[3], heatmaps.shape[2]]
        for joints in grouped_joints:
            transform = self.get_affine_transform(center, scale, heatmaps_size)
            for joint in joints:
                joint[0:2] = self.transform_point(joint[0:2], transform)
            self.transformed_joints.append(joints)
        return self.transformed_joints
