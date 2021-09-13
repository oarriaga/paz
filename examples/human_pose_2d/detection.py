import processors as pe
from paz import processors as pr
import tensorflow as tf
from pipelines import HeatmapsParser, ResizeAlignMultiScale
from pipelines import GetMultiStageOutputs, AggregateResults, FinalPrediction
from pipelines import ProcessHeatmapsTags
from dataset import JOINT_CONFIG


tag_thresh = 1
detection_thresh = 0.2
max_num_people = 30
dataset = 'COCO'
data_with_center = False
scale_factor = [1]
joint_order = JOINT_CONFIG[dataset + '_WITH_CENTER'] \
                    if data_with_center else JOINT_CONFIG[dataset]
num_joints = len(joint_order)


# preprocessing and postprocessing should be in numpy
class DetectHumanPose2D(pr.Processor):
    def __init__(self):
        super(DetectHumanPose2D, self).__init__()
        self.tf_preprocess_input = pe.TfPreprocessInput()
        self.resize_align_multi_scale = ResizeAlignMultiScale()
        self.get_multi_stage_outputs = GetMultiStageOutputs(num_joints,
                                                            dataset,
                                                            data_with_center)
        self.process_heatmaps_tags = ProcessHeatmapsTags(data_with_center)
        self.aggregate_results = AggregateResults(scale_factor)
        self.heatmaps_parser = HeatmapsParser(max_num_people, joint_order,
                                              tag_thresh, detection_thresh)
        self.get_final_preds = FinalPrediction()
        self.save_valid_image = pe.SaveValidImage(dataset)

    def call(self, model, image):
        all_preds = []
        all_scores = []
        for arg_scale in sorted(scale_factor, reverse=True):
            base_size, image_resized, center, scale = \
                    self.resize_align_multi_scale(image, arg_scale)
            image_resized = self.tf_preprocess_input(image_resized)
            image_resized = tf.expand_dims(image_resized, 0)
            heatmaps, tags = self.get_multi_stage_outputs(model, image_resized)
            heatmaps, tags = self.process_heatmaps_tags(heatmaps, tags,
                                                        base_size)
            final_heatmaps, final_tags = self.aggregate_results(heatmaps, tags,
                                                                arg_scale)
            grouped, scores = self.heatmaps_parser(final_heatmaps, final_tags)

            heatmaps_size = [final_heatmaps.get_shape()[3],
                             final_heatmaps.get_shape()[2]]
            final_results = self.get_final_preds(grouped, center, scale,
                                                 heatmaps_size)

            image = self.save_valid_image(image, final_results)

        all_preds.append(final_results)
        all_scores.append(scores)
        return all_preds, all_scores, image
