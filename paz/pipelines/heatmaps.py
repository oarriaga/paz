import numpy as np
from paz import processors as pr


class GetHeatmapsAndTags(pr.Processor):
    """Get Heatmaps and Tags from the model output.
    # Arguments
        model: Model weights trained on HigherHRNet model.
        flipped_keypoint_order: List of length 17 (number of keypoints).
            Flipped list of keypoint order.
        data_with_center: Boolean. True is the model is trained using the
            center.
        image: Numpy array. Input image of shape (H, W)

    # Returns
        heatmaps: Numpy array of shape (1, num_keypoints, H, W)
        Tags: Numpy array of shape (1, num_keypoints, H, W)
    """
    def __init__(self, model, flipped_keypoint_order, with_flip,
                 data_with_center, scale_output=True, axes=[0, 3, 1, 2]):
        super(GetHeatmapsAndTags, self).__init__()
        self.with_flip = with_flip
        self.predict = pr.SequentialProcessor(
            [pr.Predict(model), pr.TransposeOutput(axes), pr.ScaleOutput(2)])
        self.get_heatmaps = pr.GetHeatmaps(flipped_keypoint_order)
        self.get_tags = pr.GetTags(flipped_keypoint_order)
        self.postprocess = pr.SequentialProcessor()
        if data_with_center:
            self.postprocess.add(pr.RemoveLastElement())
        if scale_output:
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
