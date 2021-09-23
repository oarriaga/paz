import processors as pe
from paz import processors as pr
from pipelines import HeatmapsParser, PreprocessImage
from pipelines import GetMultiStageOutputs, AggregateResults, TransformJoints


class DetectHumanPose2D(pr.Processor):
    def __init__(self, model, max_num_people=30, dataset='COCO',
                 data_with_center=False, with_flip=True):
        super(DetectHumanPose2D, self).__init__()
        self.with_flip = with_flip
        self.preprocess_image = PreprocessImage()
        self.get_output = pr.SequentialProcessor()
        self.get_output.add(GetMultiStageOutputs(model, dataset,
                                                 data_with_center))
        self.get_output.add(AggregateResults(with_flip=self.with_flip))
        self.heatmaps_parser = HeatmapsParser(max_num_people, dataset,
                                              data_with_center)
        self.transform_joints = TransformJoints()
        self.draw_skeleton = pe.DrawSkeleton(dataset)

    def call(self, image):
        resized_image, center, scale = self.preprocess_image(image)
        heatmaps, tags = self.get_output(resized_image, self.with_flip)
        grouped_joints, scores = self.heatmaps_parser(heatmaps, tags)
        joints = self.transform_joints(grouped_joints, center, scale, heatmaps)
        image = self.draw_skeleton(image, joints)
        return joints, scores, image
