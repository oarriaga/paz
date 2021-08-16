import numpy as np
import os
import tensorflow as tf
import logging as lg
from paz import processors as pr
import processors as pe
from pathlib import Path
import time


class UpdateConfig(pr.SequentialProcessor):
    def __init__(self):
        super(UpdateConfig, self).__init__()


class CreateLogger(pr.Processor):
    def __init__(self, cfg, cfg_name, phase='train'):
        super(CreateLogger, self).__init__()
        self.cfg = cfg
        self.cfg_rank = cfg.RANK
        self.dataset = cfg.DATASET.DATASET
        self.model = cfg.MODEL.NAME
        self.cfg_name = os.path.basename(cfg_name).split('.')[0]
        self.phase = phase
        self.root_output_dir = Path(cfg.OUTPUT_DIR)
        self.log_dir = Path(cfg.LOG_DIR)
        self.create_directory = pe.CreateDirectory()
        self.replace_text = pe.ReplaceText(self.dataset)
        self.setup_logger = SetupLogger(self.cfg_rank, phase)

    def call(self):
        # removed the timer part for else condition -check for any conflict
        if not self.root_output_dir.exists() and self.cfg_rank == 0:
            self.create_directory(self.root_output_dir)
        
        self.dataset = self.replace_text(':', '_')
        final_output_dir = self.root_output_dir / self.dataset / \
                           self.model / self.cfg_name
        
        if self.cfg_rank == 0:
            self.create_directory(final_output_dir)
        
        logger, time_str = self.setup_logger(final_output_dir)
        tensorboard_log_dir = self.log_dir / self.dataset / self.model / \
                              (self.cfg_name + '_' + time_str)
        
        self.create_directory(tensorboard_log_dir)
        return logger, str(final_output_dir), str(tensorboard_log_dir)


class SetupLogger(pr.Processor):
    def __init__(self, rank, phase):
        super(SetupLogger, self).__init__()
        self.rank = rank
        self.phase = phase
        self.time_str = time.strftime('%Y-%m-%d-%H-%M')
        self.log_file = '{}_{}_rank{}.log'.format(phase, self.time_str, rank)
        self.head = '%(asctime)-15s %(message)s'

    def call(self, directory):
        final_log_file = os.path.join(directory, self.log_file)
        lg.basicConfig(filename=str(final_log_file), format=self.head)
        logger = lg.getLogger()
        logger.setLevel(lg.INFO)
        console = lg.StreamHandler()
        lg.getLogger('').addHandler(console)
        return logger, self.time_str


class Parameters():
    def __init__(self, cfg):
        super(Parameters, self).__init__()
        self.num_joints = cfg.DATASET.NUM_JOINTS
        self.max_num_people = cfg.DATASET.MAX_NUM_PEOPLE
        self.detection_threshold = cfg.TEST.DETECTION_THRESHOLD
        self.tag_threshold = cfg.TEST.TAG_THRESHOLD
        self.use_detection_val = cfg.TEST.USE_DETECTION_VAL
        self.ignore_too_much = cfg.TEST.IGNORE_TOO_MUCH
        self.pool_size = cfg.TEST.NMS_KERNEL
        self.tag_per_joint = cfg.MODEL.TAG_PER_JOINT

        if cfg.DATASET.WITH_CENTER and cfg.TEST.IGNORE_CENTER:
            self.num_joints -= 1

        if cfg.DATASET.WITH_CENTER and not cfg.TEST.IGNORE_CENTER:
            self.joint_order = [
                i-1 for i in [18, 1, 2, 3, 4, 5, 6, 7, 12,
                              13, 8, 9, 10, 11, 14, 15, 16, 17]
            ]
        else:
            self.joint_order = [
                i-1 for i in [1, 2, 3, 4, 5, 6, 7, 12, 13, 
                              8, 9, 10, 11, 14, 15, 16, 17]
            ]


class NonMaximumSuppression(pr.Processor):
    def __init__(self):
        super(NonMaximumSuppression, self).__init__
        self.transpose = pe.Transpose()
        self.pooling = pe.MaxPooling2D()
        self.check_equalities = pe.CompareElementWiseEquality()
        self.change_dtype = pe.ChangeDataType()
        self.multipy_tensor = pe.MultiplyTensors()

    def call(self, det):
        det = self.transpose(det, [0, 2, 3, 1])
        maxm = self.pooling(self.det)
        maxm = self.check_equalities(maxm, det)
        maxm = self.change_dtype(maxm, tf.float32)
        det = self.multipy_tensor(det, maxm)
        return det
        

class HeatMapParser():
    def __init__(self, cfg):
        super(HeatMapParser, self).__init__()
        self.parameters = Parameters(cfg)

    def call(self, data):




