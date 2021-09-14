import processors as pe
from paz import processors as pr
from pipelines import HeatmapsParser, ResizeAlignMultiScale
from pipelines import GetMultiStageOutputs, AggregateResults, TransformJoints
from pipelines import ProcessHeatmapsTags


class DetectHumanPose2D(pr.Processor):
    def __init__(self, max_num_people=30, dataset='COCO',
                 data_with_center=False):
        super(DetectHumanPose2D, self).__init__()
        self.resize_align_multi_scale = ResizeAlignMultiScale()
        self.preprocess = pr.SequentialProcessor()
        self.preprocess.add(pe.TfPreprocessInput())
        self.preprocess.add(pr.ExpandDims(0))
        self.get_output = pr.SequentialProcessor()
        self.get_output.add(GetMultiStageOutputs(dataset, data_with_center))
        self.get_output.add(ProcessHeatmapsTags(data_with_center))
        self.get_output.add(AggregateResults())
        self.heatmaps_parser = HeatmapsParser(max_num_people, dataset,
                                              data_with_center)
        self.transform_joints = TransformJoints()
        self.draw_skeleton = pe.DrawSkeleton(dataset)

    def call(self, model, image):
        image_resized, center, scale = self.resize_align_multi_scale(image)
        image_resized = self.preprocess(image_resized)
        heatmaps, tags = self.get_output(model, image_resized)
        grouped_joints, scores = self.heatmaps_parser(heatmaps, tags)
        joints = self.transform_joints(grouped_joints, center, scale, heatmaps)
        image = self.draw_skeleton(image, joints)
        return joints, scores, image
