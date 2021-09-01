import argparse
import os
import processors as pe
import tensorflow as tf
from pipelines import save_valid_image
from pipelines import HeatmapsParser, GetMultiScaleSize, ResizeAlignMultiScale
from pipelines import GetMultiStageOutputs, AggregateResults, FinalPrediction


num_joints = 17
joint_order = [i-1 for i in [1, 2, 3, 4, 5, 6, 7, 12,
                             13, 8, 9, 10, 11, 14, 15, 16, 17]]
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
test_with_heatmap = (True, True)
with_AE_loss = (True, False)
test_with_AE = (True, False)
dataset = 'COCO'
dataset_with_centers = False
test_ignore_centers = True
test_scale_factor = [1]
test_project2image = True
test_flip_test = True


def main():
    # load required files
    parser = argparse.ArgumentParser(description='Test keypoints network')
    parser.add_argument('-i', '--image_path', default='image',
                        help='Path to the image')
    parser.add_argument('-m', '--model_weights_path',
                        default='models_weights_tf',
                        help='Path to the model weights')
    args = parser.parse_args()

    image_path = os.path.join(args.image_path, 'image2.jpg')
    model_path = os.path.join(args.model_weights_path, 'HigherHRNet')

    load_image = pe.LoadImage()
    load_model = pe.LoadModel()
    model = load_model(model_path)
    image = load_image(image_path)
    # print(model.summary())
    print("\n==> Model loaded!\n")

    heatmaps_parser = HeatmapsParser(num_joints, joint_order, detection_thresh,
                                     max_num_people, ignore_too_much,
                                     use_detection_val, tag_thresh,
                                     tag_per_joint)

    get_multi_scale_size = GetMultiScaleSize(input_size, min_scale)
    base_size, center, scale = get_multi_scale_size(image, current_scale)
    resize_align_multi_scale = ResizeAlignMultiScale(input_size, min_scale)
    get_multi_stage_outputs = GetMultiStageOutputs(with_heatmap_loss,
                                                   test_with_heatmap,
                                                   with_AE_loss,
                                                   test_with_AE,
                                                   dataset,
                                                   dataset_with_centers,
                                                   tag_per_joint,
                                                   test_ignore_centers,
                                                   test_scale_factor,
                                                   test_project2image,
                                                   test_flip_test,
                                                   num_joints,
                                                   test_flip_test,
                                                   test_project2image)

    aggregate_results = AggregateResults(test_scale_factor, test_project2image,
                                         test_flip_test)
    get_final_preds = FinalPrediction()

    all_preds = []
    all_scores = []
    for idx, s in enumerate(sorted(test_scale_factor, reverse=True)):
        image_resized, center, scale = resize_align_multi_scale(image, s)
        image_resized = tf.keras.applications.imagenet_utils.preprocess_input(image_resized,
                                                                              data_format=None, mode='torch')

        image_resized = tf.expand_dims(image_resized, 0)
        heatmaps, tags = get_multi_stage_outputs(model, image_resized,
                                                 base_size)

        final_heatmaps, final_tags = aggregate_results(s, heatmaps, tags)
        grouped, scores = heatmaps_parser(final_heatmaps, final_tags)

        heatmaps_size = [final_heatmaps.get_shape()[3],
                         final_heatmaps.get_shape()[2]]

        final_results = get_final_preds(grouped, center, scale, heatmaps_size)

        prefix = '{}_'.format(os.path.join('output', 'result_valid_3'))
        save_valid_image(image, final_results, '{}.jpg'.format(prefix),
                         dataset='COCO')
        print(f"image {'{}.jpg'.format(prefix)} saved!")

        all_preds.append(final_results)
        all_scores.append(scores)


if __name__ == '__main__':
    main()
