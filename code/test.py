import argparse
import logging
import os

import torch
import torch.utils.data

from config import get_default_config

from inference import do_evaluation
from models.ssd_detector import SSDDetector
from utils.checkpoint import CheckPointer
from utils.logger import setup_logger


def evaluation(cfg, ckpt):
    logger = logging.getLogger("SSD.inference")

    model = SSDDetector(cfg)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR, logger=logger)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    do_evaluation(cfg, model, sample_rate=0.1)


def main():
    parser = argparse.ArgumentParser(description='SSD Evaluation on VOC and COCO dataset.')

    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
        type=str,
    )
    parser.add_argument("--output_dir", default="eval_results", type=str, help="The directory to store evaluation results.")
    args = parser.parse_args()

    if torch.cuda.is_available():
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True

    cfg = get_default_config()
    cfg.freeze()

    logger = setup_logger("SSD", 0, cfg.OUTPUT_DIR)
    logger.info(args)

    logger.info("Running with config:\n{}".format(cfg))
    evaluation(cfg, ckpt=args.ckpt)


if __name__ == '__main__':
    main()
