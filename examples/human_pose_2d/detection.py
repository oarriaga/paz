import processors as pe
from paz import processors as pr
from pipelines import HeatmapsParser, PreprocessImage
from pipelines import GetMultiStageOutputs, AggregateResults, TransformJoints


class DetectHumanPose2D(pr.Processor):
    def __init__(self, model, data_info, max_num_people=30, with_flip=True):
        super(DetectHumanPose2D, self).__init__()
        self.with_flip = with_flip
        self.preprocess_image = PreprocessImage()
        self.get_output = pr.SequentialProcessor(
            [GetMultiStageOutputs(model, data_info),
             AggregateResults(with_flip=self.with_flip)])
        self.heatmaps_parser = HeatmapsParser(max_num_people, data_info)
        self.transform_joints = TransformJoints()
        self.draw_skeleton = pe.DrawSkeleton(data_info)

    def call(self, image):
        resized_image, center, scale = self.preprocess_image(image)
        heatmaps, tags = self.get_output(resized_image, self.with_flip)
        grouped_joints, scores = self.heatmaps_parser(heatmaps, tags)
        joints = self.transform_joints(grouped_joints, center, scale, heatmaps)
        image = self.draw_skeleton(image, joints)
        return joints, scores, image
