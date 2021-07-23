# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import torch
import tensorflow as tf

import sys
sys.path.append(r".../HigherHRNet-TF2.0")

from lib.config import cfg
from lib.config import update_config

# set all seeds
np.random.seed(999)
torch.manual_seed(999)
tf.random.set_seed(999)

logger = logging.getLogger(__name__)


def make_input(t, need_cuda=False):
    inp = tf.convert_to_tensor(t)
    inp = tf.math.reduce_sum(inp)
    if need_cuda:
        inp = inp.cuda()
    return inp


class HeatmapLoss(tf.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, pred, gt, mask):
        assert pred.shape == gt.shape
        loss = ((pred - gt)**2) * tf.broadcast_to(mask[:, None, :, :], pred.shape)
        l1 = tf.math.reduce_mean(loss, axis=3)
        l2 = tf.math.reduce_mean(l1, axis=2)
        loss = tf.math.reduce_mean(l2, axis=1)
        # loss = loss.mean(dim=3).mean(dim=2).mean(dim=1)
        return loss


class AELoss(tf.Module):
    def __init__(self, loss_type):
        super().__init__()
        self.loss_type = loss_type

    def singleTagLoss(self, pred_tag, joints):   # refer Lg equation from AE paper
        """
        associative embedding loss for one image
        """
        tags = []
        pull = 0
        for joints_per_person in joints:
            tmp = []
            for joint in joints_per_person:
                if joint[1] > 0:
                    tmp.append(pred_tag[joint[0]])
            if len(tmp) == 0:
                continue
            tmp = tf.stack(tmp)
            tags.append(tf.math.reduce_mean(tmp, axis=0))
            pull = pull + tf.math.reduce_mean((tmp - tf.broadcast_to(tags[-1], tmp.shape))**2)

        num_tags = len(tags)
        if num_tags == 0:
            return make_input(tf.zeros(1, dtype=tf.float64)), \
                make_input(tf.zeros(1, dtype=tf.float64))
        elif num_tags == 1:
            return make_input(tf.zeros(1, dtype=tf.float64)), \
                pull/(num_tags)

        tags = tf.stack(tags)

        size = (num_tags, num_tags)
        A = tf.broadcast_to(tags, size)
        B = tf.transpose(A, perm=[1, 0])

        diff = A - B
        if self.loss_type == 'exp':
            diff = tf.math.pow(diff, 2)
            push = tf.math.exp(-diff)
            push = tf.math.reduce_sum(push) - num_tags
        elif self.loss_type == 'max':
            diff = 1 - tf.math.abs(diff)
            push = tf.clip_by_value(diff, clip_value_min=0).sum() - num_tags
        else:
            raise ValueError('Unknown ae loss type')

        return push/((num_tags - 1) * num_tags) * 0.5, \
            pull/(num_tags)

    def __call__(self, tags, joints):
        """
        accumulate the tag loss for each image in the batch
        """
        pushes, pulls = [], []
        joints = joints.numpy()
        batch_size = tags.get_shape()[0]
        for i in range(batch_size):
            push, pull = self.singleTagLoss(tags[i], joints[i])
            pushes.append(push)
            pulls.append(pull)
        return tf.stack(pushes), tf.stack(pulls)


class JointsMSELoss(tf.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = tf.keras.losses.MeanSquaredError()
        self.use_target_weight = use_target_weight

    def __call__(self, output, target, target_weight):
        batch_size = output.get_shape()[0]
        num_joints = output.get_shape()[1]
        heatmaps_pred = tf.split(tf.reshape(output, (batch_size, num_joints, -1)), num_or_size_splits=2, axis=1)
        heatmaps_gt = tf.split(tf.reshape(target, (batch_size, num_joints, -1)), 2, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = tf.squeeze(heatmaps_pred[idx])
            heatmap_gt = tf.squeeze(heatmaps_gt[idx])
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    tf.math.multiply(heatmap_pred, target_weight[:, idx]),
                    tf.math.multiply(heatmap_gt, target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class MultiLossFactory(tf.Module):
    def __init__(self, cfg):
        super().__init__()
        # init check
        self._init_check(cfg)

        self.num_joints = 3 #cfg.MODEL.NUM_JOINTS
        self.num_stages = cfg.LOSS.NUM_STAGES

        self.heatmaps_loss = \
                [
                    HeatmapLoss()
                    if with_heatmaps_loss else None
                    for with_heatmaps_loss in cfg.LOSS.WITH_HEATMAPS_LOSS
                ]
        self.heatmaps_loss_factor = cfg.LOSS.HEATMAPS_LOSS_FACTOR

        self.ae_loss = \
                [
                    AELoss(cfg.LOSS.AE_LOSS_TYPE) if with_ae_loss else None
                    for with_ae_loss in cfg.LOSS.WITH_AE_LOSS
                ]
        self.push_loss_factor = cfg.LOSS.PUSH_LOSS_FACTOR
        self.pull_loss_factor = cfg.LOSS.PULL_LOSS_FACTOR

    def __call__(self, outputs, heatmaps, masks, joints):
        # forward check
        self._forward_check(outputs, heatmaps, masks, joints)

        heatmaps_losses = []
        push_losses = []
        pull_losses = []
        for idx in range(len(outputs)):
            offset_feat = 0
            if self.heatmaps_loss[idx]:
                heatmaps_pred = outputs[idx][:, :self.num_joints]
                offset_feat = self.num_joints

                heatmaps_loss = self.heatmaps_loss[idx](
                    heatmaps_pred, heatmaps[idx], masks[idx]
                )
                heatmaps_loss = heatmaps_loss * self.heatmaps_loss_factor[idx]
                heatmaps_losses.append(heatmaps_loss)
            else:
                heatmaps_losses.append(None)

            if self.ae_loss[idx]:
                tags_pred = outputs[idx][:, offset_feat:]
                batch_size = tags_pred.get_shape()[0]
                tags_pred = tf.reshape(tags_pred, [batch_size, -1, 1])

                push_loss, pull_loss = self.ae_loss[idx](
                    tags_pred, joints[idx]
                )
                push_loss = push_loss * self.push_loss_factor[idx]
                pull_loss = pull_loss * self.pull_loss_factor[idx]

                push_losses.append(push_loss)
                pull_losses.append(pull_loss)
            else:
                push_losses.append(None)
                pull_losses.append(None)

        return heatmaps_losses, push_losses, pull_losses

    def _init_check(self, cfg):
        assert isinstance(cfg.LOSS.WITH_HEATMAPS_LOSS, (list, tuple)), \
            'LOSS.WITH_HEATMAPS_LOSS should be a list or tuple'
        assert isinstance(cfg.LOSS.HEATMAPS_LOSS_FACTOR, (list, tuple)), \
            'LOSS.HEATMAPS_LOSS_FACTOR should be a list or tuple'
        assert isinstance(cfg.LOSS.WITH_AE_LOSS, (list, tuple)), \
            'LOSS.WITH_AE_LOSS should be a list or tuple'
        assert isinstance(cfg.LOSS.PUSH_LOSS_FACTOR, (list, tuple)), \
            'LOSS.PUSH_LOSS_FACTOR should be a list or tuple'
        assert isinstance(cfg.LOSS.PUSH_LOSS_FACTOR, (list, tuple)), \
            'LOSS.PUSH_LOSS_FACTOR should be a list or tuple'
        assert len(cfg.LOSS.WITH_HEATMAPS_LOSS) == cfg.LOSS.NUM_STAGES, \
            'LOSS.WITH_HEATMAPS_LOSS and LOSS.NUM_STAGE should have same length, got {} vs {}.'.\
                format(len(cfg.LOSS.WITH_HEATMAPS_LOSS), cfg.LOSS.NUM_STAGES)
        assert len(cfg.LOSS.WITH_HEATMAPS_LOSS) == len(cfg.LOSS.HEATMAPS_LOSS_FACTOR), \
            'LOSS.WITH_HEATMAPS_LOSS and LOSS.HEATMAPS_LOSS_FACTOR should have same length, got {} vs {}.'.\
                format(len(cfg.LOSS.WITH_HEATMAPS_LOSS), len(cfg.LOSS.HEATMAPS_LOSS_FACTOR))
        assert len(cfg.LOSS.WITH_AE_LOSS) == cfg.LOSS.NUM_STAGES, \
            'LOSS.WITH_AE_LOSS and LOSS.NUM_STAGE should have same length, got {} vs {}.'.\
                format(len(cfg.LOSS.WITH_AE_LOSS), cfg.LOSS.NUM_STAGES)
        assert len(cfg.LOSS.WITH_AE_LOSS) == len(cfg.LOSS.PUSH_LOSS_FACTOR), \
            'LOSS.WITH_AE_LOSS and LOSS.PUSH_LOSS_FACTOR should have same length, got {} vs {}.'. \
                format(len(cfg.LOSS.WITH_AE_LOSS), len(cfg.LOSS.PUSH_LOSS_FACTOR))
        assert len(cfg.LOSS.WITH_AE_LOSS) == len(cfg.LOSS.PULL_LOSS_FACTOR), \
            'LOSS.WITH_AE_LOSS and LOSS.PULL_LOSS_FACTOR should have same length, got {} vs {}.'. \
                format(len(cfg.LOSS.WITH_AE_LOSS), len(cfg.LOSS.PULL_LOSS_FACTOR))

    def _forward_check(self, outputs, heatmaps, masks, joints):
        assert isinstance(outputs, list), \
            'outputs should be a list, got {} instead.'.format(type(outputs))
        assert isinstance(heatmaps, list), \
            'heatmaps should be a list, got {} instead.'.format(type(heatmaps))
        assert isinstance(masks, list), \
            'masks should be a list, got {} instead.'.format(type(masks))
        assert isinstance(joints, list), \
            'joints should be a list, got {} instead.'.format(type(joints))
        assert len(outputs) == self.num_stages, \
            'len(outputs) and num_stages should been same, got {} vs {}.'.format(len(outputs), self.num_stages)
        assert len(outputs) == len(heatmaps), \
            'outputs and heatmaps should have same length, got {} vs {}.'.format(len(outputs), len(heatmaps))
        assert len(outputs) == len(masks), \
            'outputs and masks should have same length, got {} vs {}.'.format(len(outputs), len(masks))
        assert len(outputs) == len(joints), \
            'outputs and joints should have same length, got {} vs {}.'.format(len(outputs), len(joints))
        assert len(outputs) == len(self.heatmaps_loss), \
            'outputs and heatmaps_loss should have same length, got {} vs {}.'. \
                format(len(outputs), len(self.heatmaps_loss))
        assert len(outputs) == len(self.ae_loss), \
            'outputs and ae_loss should have same length, got {} vs {}.'. \
                format(len(outputs), len(self.ae_loss))


def test_ae_loss():
    import numpy as np
    t = tf.convert_to_tensor(np.arange(0, 32).reshape(1, 2, 4, 4).astype(np.float)*0.1)

    ae_loss = AELoss(loss_type='exp')

    joints = np.zeros((2, 2, 2))
    joints[0, 0] = (3, 1)
    joints[1, 0] = (10, 1)
    joints[0, 1] = (22, 1)
    joints[1, 1] = (30, 1)

    joints = tf.cast(joints, tf.int64)
    joints = tf.reshape(joints, [1, 2, 2, 2])
    t = tf.reshape(t, [1, -1, 1])

    l = ae_loss(t, joints)


def test_heatmap_loss():
    import numpy as np
    pred = tf.convert_to_tensor(np.arange(0, 32).reshape(2, 2, 2, 4).astype(np.float)*0.1)
    gt = tf.convert_to_tensor(np.arange(0, 32).reshape(2, 2, 2, 4).astype(np.float)*0.12)
    mask = tf.convert_to_tensor(np.arange(0, 16).reshape(2, 2, 4).astype(np.float)*0.09)

    x = tf.broadcast_to(mask[:, None, :, :], pred.shape)

    heatmap_loss = HeatmapLoss()
    l = heatmap_loss(pred, gt, mask)


def test_joint_mse_loss():
    import numpy as np
    output = tf.convert_to_tensor(np.arange(0, 32).reshape(2, 2, 2, 4).astype(np.float) * 0.1)
    target = tf.convert_to_tensor(np.arange(0, 32).reshape(2, 2, 2, 4).astype(np.float) * 0.12)
    target_weight = tf.convert_to_tensor(np.arange(0, 32).reshape(8, 4).astype(np.float) * 0.09)

    joint_mse_loss = JointsMSELoss(use_target_weight=True)

    l = joint_mse_loss(output, target, target_weight)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    return args


def test_loss_factory():
    import numpy as np

    args = parse_args()
    update_config(cfg, args)

    outputs = [tf.convert_to_tensor(np.random.rand(5, 6, 2, 2)), \
               tf.convert_to_tensor(np.random.rand(5, 3, 4, 4))]
    heatmaps = [tf.convert_to_tensor(np.random.rand(5, 3, 2, 2)), \
               tf.convert_to_tensor(np.random.rand(5, 3, 4, 4))]
    masks = [tf.convert_to_tensor(np.random.rand(5, 2, 2)), \
               tf.convert_to_tensor(np.random.rand(5, 4, 4))]
    joints = [tf.convert_to_tensor(np.ndarray.astype(np.random.rand(5, 2, 3, 2)*2, int)), \
               tf.convert_to_tensor(np.ndarray.astype(np.random.rand(5, 2, 3, 2)*2, int))]

    loss_factory = MultiLossFactory(cfg)

    l1, l2, l3 = loss_factory(outputs, heatmaps, masks, joints)


if __name__ == '__main__':
    test_loss_factory()
