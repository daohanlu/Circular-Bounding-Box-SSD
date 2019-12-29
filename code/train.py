import os
import logging
import torch

from utils.checkpoint import CheckPointer
from models.ssd_detector import SSDDetector
from learning_rate.build import make_lr_scheduler, make_optimizer
from training.trainer import train_ssd_detector
from inference import do_evaluation
from data.loader import make_data_loader
from config import get_default_config
from utils.logger import setup_logger


def train(cfg, model):
    logger = logging.getLogger('SSD.trainer')
    device = torch.cuda.current_device()
    model.to(device)

    lr = cfg.SOLVER.LR
    optimizer = make_optimizer(cfg, model, lr)

    milestones = [step for step in cfg.SOLVER.LR_STEPS]
    scheduler = make_lr_scheduler(cfg, optimizer, milestones)

    arguments = {"iteration": 0}
    save_to_disk = True
    checkpointer = CheckPointer(model, optimizer, scheduler, cfg.OUTPUT_DIR, save_to_disk, logger)
    checkpoint_data = checkpointer.load()
    arguments.update(checkpoint_data)

    max_iter = cfg.SOLVER.MAX_ITER  # should be divided by num GPUs but right now multi gpu is not supported
    train_loader = make_data_loader(cfg, is_train=True, max_iter=max_iter,
                                    start_iter=arguments['iteration'])

    model = train_ssd_detector(cfg, model, train_loader, optimizer, scheduler, checkpointer, device, arguments)
    return model


def main():
    cfg = get_default_config()
    logger = setup_logger("SSD", 0, cfg.OUTPUT_DIR)

    model = SSDDetector(cfg)
    train(cfg, model)
    logger.info('Start evaluating...')
    torch.cuda.empty_cache()  # speed up evaluating after training finished
    do_evaluation(cfg, model)


if __name__ == '__main__':
    main()
