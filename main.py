import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.nn as nn
from logger import create_logger

from experiments.exp_port import Exp_Forecast
from config import get_config
from parse import parse_option
import pdb

import warnings
warnings.filterwarnings('ignore')

def main():

    args, config = parse_option()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")

    #print config
    # logger.info(config.dump())

    # Start experiments
    Exp = Exp_Forecast(args, config, logger)

    # Exp._get_data("train")
    Exp.train("testing")

if __name__ == "__main__":
    main()

