import argparse
import os
import numpy as np
import cv2
import processors as pe
import tensorflow as tf
from pipelines import HeatmapsParser, ResizeAlignMultiScale, draw_skeleton
from pipelines import GetMultiStageOutputs, AggregateResults, FinalPrediction
from tensorflow.keras.applications import imagenet_utils


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

    model_path = os.path.join(args.model_weights_path, 'HigherHRNet')

    load_model = pe.LoadModel()
    model = load_model(model_path)
    # print(model.summary())
    print("\n==> Model loaded!\n")

    resize_align_multi_scale = ResizeAlignMultiScale()
    get_multi_stage_outputs = GetMultiStageOutputs(with_heatmap_loss,
                                                   test_with_heatmap,
                                                   with_AE_loss,
                                                   test_with_AE,
                                                   dataset_with_centers,
                                                   tag_per_joint,
                                                   test_ignore_centers,
                                                   num_joints,
                                                   test_flip_test,
                                                   test_project2image)

    aggregate_results = AggregateResults(test_scale_factor, test_project2image,
                                         test_flip_test)
    heatmaps_parser = HeatmapsParser()
    get_final_preds = FinalPrediction()
    video_capture = cv2.VideoCapture(0)

    all_preds = []
    all_scores = []
    while True:
        final_heatmaps = None
        # Capture frame-by-frame
        ret, oriImg = video_capture.read()
        oriImg = np.array(oriImg).astype(np.uint8)

        base_size, image_resized, center, scale = resize_align_multi_scale(oriImg, 1)
        image_resized = imagenet_utils.preprocess_input(image_resized,
                                                        data_format=None,
                                                        mode='torch')

        image_resized = tf.expand_dims(image_resized, 0)

        heatmaps, tags = get_multi_stage_outputs(model, image_resized,
                                                 base_size)

        final_heatmaps, final_tags = aggregate_results(1, heatmaps, tags)
        grouped, scores = heatmaps_parser(final_heatmaps, final_tags)

        heatmaps_size = [final_heatmaps.get_shape()[3],
                         final_heatmaps.get_shape()[2]]

        final_results = get_final_preds(grouped, center, scale, heatmaps_size)

        all_preds.append(final_results)
        all_scores.append(scores)

        out = draw_skeleton(oriImg, final_results, dataset='COCO')
        # Display the resulting frame
        cv2.imshow('Video', out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
