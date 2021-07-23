# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from dataset.transforms import FLIP_CONFIG

import _init_paths


def get_multi_stage_outputs(cfg, model, image, with_flip=False, project2image=False, size_projected=None):
    heatmaps_avg = 0
    num_heatmaps = 0
    heatmaps = []
    tags = []

    outputs = model(image)

    for i, output in enumerate(outputs):  # len(outputs) = 2, first one is 34 channels image, second is 17 channels
        if len(outputs) > 1 and i != len(outputs) - 1:
            output = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(output)

        offset_feat = cfg.DATASET.NUM_JOINTS if cfg.LOSS.WITH_HEATMAPS_LOSS[i] else 0

        if cfg.LOSS.WITH_HEATMAPS_LOSS[i] and cfg.TEST.WITH_HEATMAPS[i]:
            heatmaps_avg += output[:, :, :, :cfg.DATASET.NUM_JOINTS]
            num_heatmaps += 1

        if cfg.LOSS.WITH_AE_LOSS[i] and cfg.TEST.WITH_AE[i]:
            tags.append(output[:, :, :, offset_feat:])

    if num_heatmaps > 0:
        heatmaps.append(heatmaps_avg/num_heatmaps)

    if with_flip:
        if 'coco' in cfg.DATASET.DATASET:
            dataset_name = 'COCO'
        elif 'crowd_pose' in cfg.DATASET.DATASET:
            dataset_name = 'CROWDPOSE'
        else:
            raise ValueError('Please implement flip_index for new dataset: %s.' % cfg.DATASET.DATASET)
        flip_index = FLIP_CONFIG[dataset_name + '_WITH_CENTER'] \
            if cfg.DATASET.WITH_CENTER else FLIP_CONFIG[dataset_name]

        heatmaps_avg = 0
        num_heatmaps = 0
        outputs_flip = model(tf.reverse(image, [2]))  # flip about last axis is mirror flip about each row
        for i in range(len(outputs_flip)):
            output = outputs_flip[i]
            if len(outputs_flip) > 1 and i != len(outputs_flip) - 1:
                output = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(output)
            output = tf.reverse(output, [2])  # unflip
            outputs.append(output)

            offset_feat = cfg.DATASET.NUM_JOINTS if cfg.LOSS.WITH_HEATMAPS_LOSS[i] else 0

            if cfg.LOSS.WITH_HEATMAPS_LOSS[i] and cfg.TEST.WITH_HEATMAPS[i]:
                temp = tf.gather(output[:, :, :, :cfg.DATASET.NUM_JOINTS], indices=flip_index, axis=-1)
                heatmaps_avg += temp
                num_heatmaps += 1

            if cfg.LOSS.WITH_AE_LOSS[i] and cfg.TEST.WITH_AE[i]:
                tags.append(output[:, :, :, offset_feat:])
                if cfg.MODEL.TAG_PER_JOINT:
                    tags[-1] = tf.gather(tags[-1], indices=flip_index, axis=-1)
        heatmaps.append(heatmaps_avg/num_heatmaps)
        # heatmap[0] torch.Size([1, 17, 256, w]) | heatmap[1] torch.Size([1, 17, 256, w])

    if cfg.DATASET.WITH_CENTER and cfg.TEST.IGNORE_CENTER:
        heatmaps = [hms[:, :-1] for hms in heatmaps]
        tags = [tms[:, :-1] for tms in tags]

    if project2image and size_projected:
        heatmaps = [
            tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(hms)  # upsample heatmaps
            for hms in heatmaps
        ]

        tags = [
            tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(tms)
            for tms in tags
        ]

    return outputs, heatmaps, tags


def aggregate_results(
        cfg, scale_factor, final_heatmaps, tags_list, heatmaps, tags
):
    if scale_factor == 1 or len(cfg.TEST.SCALE_FACTOR) == 1:
        if final_heatmaps is not None and not cfg.TEST.PROJECT2IMAGE:
            tags = [
                tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(tms)
                for tms in tags
            ]
        for tms in tags:
            tags_list.append(tf.expand_dims(tms, axis=-1))
    heatmaps_avg = (heatmaps[0] + heatmaps[1])/2.0 if cfg.TEST.FLIP_TEST else heatmaps[0]

    if final_heatmaps is None:
        final_heatmaps = heatmaps_avg
    elif cfg.TEST.PROJECT2IMAGE:
        final_heatmaps += heatmaps_avg
    else:
        final_heatmaps += tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(heatmaps_avg)
    return final_heatmaps, tags_list
