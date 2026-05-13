"""utils for logging
"""
import logging
from typing import *
import os
import sys
from torch.utils.tensorboard import SummaryWriter

def resetLogger(logdir:Optional[str]=None, fname:Optional[str]=None):
    """getting logger for a job

    logs will be displayed on stdout as well as in `logdir` if present

    Args:
        job_name (str): name of the job
        logdir (str, optional): path to save the logidr. 
            If None, logs will be output to stdout. Defaults to None.

    Returns:
        logging.Logger: logger for the job
    """
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)

    logger.addHandler(logging.StreamHandler(sys.stderr))

    if logdir is not None:
        assert fname is not None, logger.error("fname has to be set")
        os.makedirs(logdir,exist_ok=True)
        logger.addHandler(logging.FileHandler(os.path.join(logdir, fname)))

def getSummaryWriter(tb_dir:str):
    """getting tensorboard summary writer for a job

    Args:
        job_name (str): name of the job
        logdir (str, optional): path to save the tensorboard summary. 
        Defaults to "log".

    Returns:
        SummaryWriter: tensorboard summary writer for the job
    """
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(tb_dir)
    return writer  