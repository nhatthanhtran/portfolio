import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.nn as nn
from logger import create_logger

from experiments.exp_port import Exp_Forecast


import warnings
warnings.filterwarnings('ignore')

def parse_option():
    parser = argparse.ArgumentParser('Portfolio training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--pretrained', type=str, help='Finetune 384 initial checkpoint.', default='')
    parser.add_argument('--find-unused-params', action='store_true', default=False)

    # Model Exponential Moving Average

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main()
    args, config = parse_option()

    

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, 1, name=f"{config.MODEL.NAME}")

    #print config
    logger.info(config.dump())

    # Start experiments
    Exp = Exp_Forecast(args, configs, logger)

if __name__ == "__main__":
    main()

