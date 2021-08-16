import argparse
import os
import numpy as np
from lib.config import cfg, check_config, update_config
import processors as pe
from paz.backend.image import load_image
import tensorflow as tf
from pipelines import CreateLogger


if __name__ == '__main__':
    # load required files
    parser = argparse.ArgumentParser(description='Test keypoints network')
    parser.add_argument('-c', '--cfg', type=str, 
                        default='lib/config/w32_512_adam_lr1e-3.yaml',
                        help='Path to the config file')
    parser.add_argument('-i', '--image_path', default='image',
                        help='Path to the image')
    parser.add_argument('-m', '--model_weights_path', 
                        default='models_weights_tf',
                        help='Path to the model weights')
    args = parser.parse_args()

    # config_path = os.path.join(args.cfg_path, 'w32_512_adam_lr1e-3.yaml')
    image_path = os.path.join(args.image_path, 'img.jpg')
    model_path = os.path.join(args.model_weights_path, 'HigherHRNet')

    image = load_image(image_path)
    load_model = pe.LoadModel() 
    # model = load_model(model_path)

    # preprocessing config files
    update_config(cfg, args)
    check_config(cfg)
    create_logger = CreateLogger(cfg, args.cfg, 'valid')
    logger, final_output_dir, tb_log_dir = create_logger()
    print('done')







