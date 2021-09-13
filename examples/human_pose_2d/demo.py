import argparse
import os
import processors as pe
import tensorflow as tf
from pipelines import HeatmapsParser, ResizeAlignMultiScale
from pipelines import GetMultiStageOutputs, AggregateResults, FinalPrediction
from pipelines import PreprocessHeatmapsTags
from dataset import JOINT_CONFIG


tag_thresh = 1
detection_thresh = 0.2
max_num_people = 30
ignore_too_much = False
use_detection_val = True
tag_per_joint = True
input_size = 512
current_scale = 1
min_scale = 1

with_heatmap_loss = (True, True)
with_heatmap = (True, True)
with_AE_loss = (True, False)
with_AE = (True, False)
dataset = 'COCO'
dataset_with_centers = False
ignore_centers = True
scale_factor = [1]
project2image = True
flip_test = True

joint_order = JOINT_CONFIG[dataset + '_WITH_CENTER'] \
                    if dataset_with_centers else JOINT_CONFIG[dataset]
num_joints = len(joint_order)


def main():
    # load required files
    parser = argparse.ArgumentParser(description='Test keypoints network')
    parser.add_argument('-i', '--image_path', default='image',
                        help='Path to the image')
    parser.add_argument('-m', '--model_weights_path',
                        default='models_weights_tf',
                        help='Path to the model weights')
    args = parser.parse_args()

    image_path = os.path.join(args.image_path, 'image1.jpg')
    model_path = os.path.join(args.model_weights_path, 'HigherHRNet')

    load_image = pe.LoadImage()
    load_model = pe.LoadModel()
    model = load_model(model_path)
    image = load_image(image_path)
    tf_preprocess_input = pe.TfPreprocessInput()
    # print(model.summary())
    print("\n==> Model loaded!\n")

    resize_align_multi_scale = ResizeAlignMultiScale()
    get_multi_stage_outputs = GetMultiStageOutputs(with_heatmap_loss,
                                                   with_heatmap, with_AE_loss,
                                                   with_AE, tag_per_joint,
                                                   dataset_with_centers,
                                                   num_joints, flip_test)

    preprocess_heatmaps_tags = PreprocessHeatmapsTags(dataset_with_centers,
                                                      ignore_centers,
                                                      project2image)
    aggregate_results = AggregateResults(scale_factor, project2image,
                                         flip_test)
    heatmaps_parser = HeatmapsParser()
    get_final_preds = FinalPrediction()
    save_valid_image = pe.SaveValidImage(dataset)

    all_preds = []
    all_scores = []
    for idx, s in enumerate(sorted(scale_factor, reverse=True)):
        base_size, image_resized, center, scale = resize_align_multi_scale(image, s)
        image_resized = tf_preprocess_input(image_resized)
        image_resized = tf.expand_dims(image_resized, 0)

        heatmaps, tags = get_multi_stage_outputs(model, image_resized)
        heatmaps, tags = preprocess_heatmaps_tags(heatmaps, tags, base_size)
        final_heatmaps, final_tags = aggregate_results(heatmaps, tags, s)
        grouped, scores = heatmaps_parser(final_heatmaps, final_tags)

        heatmaps_size = [final_heatmaps.get_shape()[3],
                         final_heatmaps.get_shape()[2]]

        final_results = get_final_preds(grouped, center, scale, heatmaps_size)

        prefix = '{}'.format(os.path.join('output', 'result'))
        save_valid_image(image, final_results, '{}.jpg'.format(prefix))
        print(f"image {'{}.jpg'.format(prefix)} saved!")

        all_preds.append(final_results)
        all_scores.append(scores)


if __name__ == '__main__':
    main()
