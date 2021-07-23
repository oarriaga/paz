# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import numpy as np
import PIL

import tensorflow as tf

import _init_paths

from config import cfg
from config import check_config
from config import update_config
from core.inference import get_multi_stage_outputs
from core.inference import aggregate_results
from core.group import HeatmapParser
from utils.utils import create_logger
from utils.vis import save_valid_image
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size


def parse_args():
    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(cfg, args)
    check_config(cfg)
    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'valid')
    logger.info(pprint.pformat(args))

    model = tf.keras.models.load_model('.../models_weights_tf/HigherHRNet')
    # model.summary()
    print(f"\n==> Model loaded!\n")

    img_path = '.../img.jpg'
    img = np.array(PIL.Image.open(img_path, 'r')).astype(np.uint8)

    parser = HeatmapParser(cfg)
    all_preds = []
    all_scores = []

    # size at scale 1.0
    base_size, center, scale = get_multi_scale_size(img, cfg.DATASET.INPUT_SIZE, 1.0, min(cfg.TEST.SCALE_FACTOR))
    # height after resizing is 512 always

    final_heatmaps = None
    tags_list = []
    for idx, s in enumerate(sorted(cfg.TEST.SCALE_FACTOR, reverse=True)):
        input_size = cfg.DATASET.INPUT_SIZE
        image_resized, center, scale = resize_align_multi_scale(img, input_size, s, min(cfg.TEST.SCALE_FACTOR))
        image_resized = tf.keras.applications.imagenet_utils.preprocess_input(image_resized,
                                                                              data_format=None, mode='torch')
        image_resized = tf.expand_dims(image_resized, 0)
        outputs, heatmaps, tags = get_multi_stage_outputs(
            cfg, model, image_resized, cfg.TEST.FLIP_TEST,
            cfg.TEST.PROJECT2IMAGE, base_size
        )
       
        final_heatmaps, tags_list = aggregate_results(cfg, s, final_heatmaps, tags_list, heatmaps, tags)
        final_heatmaps = final_heatmaps / float(len(cfg.TEST.SCALE_FACTOR))
        final_heatmaps = tf.transpose(final_heatmaps, [0, 3, 1, 2])
        tags = tf.concat(tags_list, axis=4)  # makes size [1, 17, 512, w, 2]
        tags = tf.transpose(tags, [0, 3, 1, 2, 4])
        grouped, scores = parser.parse(final_heatmaps, tags, cfg.TEST.ADJUST, cfg.TEST.REFINE)
        final_results = get_final_preds(grouped, center, scale,
                                        [final_heatmaps.get_shape()[3], final_heatmaps.get_shape()[2]]
                                        )                        

        prefix = '{}_'.format(os.path.join(final_output_dir, 'result_valid_3'))
        save_valid_image(img, final_results, '{}.jpg'.format(prefix), dataset='COCO')
        print(f"image {'{}.jpg'.format(prefix)} saved!")

        all_preds.append(final_results)            
        all_scores.append(scores)                  


if __name__ == '__main__':
    main()
