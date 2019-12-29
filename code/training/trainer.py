import collections
import datetime
import logging
import os
import time

import torch
import tensorboardX

from inference import do_evaluation
from utils.metric_logger import MetricLogger


def write_metric(eval_result, prefix, summary_writer, global_step):
    for key in eval_result:
        value = eval_result[key]
        tag = '{}/{}'.format(prefix, key)
        if isinstance(value, collections.Mapping):
            write_metric(value, tag, summary_writer, global_step)
        else:
            summary_writer.add_scalar(tag, value, global_step=global_step)


def train_ssd_detector(cfg, model,
                       data_loader,
                       optimizer,
                       scheduler,
                       checkpointer,
                       device,
                       arguments, ):
    logger = logging.getLogger("SSD.trainer")
    logger.info("Start training ...")
    meters = MetricLogger()
    summary_writer = tensorboardX.SummaryWriter(log_dir=os.path.join(cfg.OUTPUT_DIR, 'tf_logs'))

    model.train()
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        # offset iteration by 1 to make it start at 1
        iteration = iteration + 1
        arguments["iteration"] = iteration
        scheduler.step()

        images = images.to(device)
        targets = targets.to(device)
        loss_dict = model(images, targets=targets)
        loss = sum(loss for loss in loss_dict.values())
        meters.update(total_loss=loss, **loss_dict)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time)
        if iteration % cfg.SOLVER.LOG_STEP == 0:
            # print('Logging iteration {}'.format(iteration))
            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                meters.delimiter.join([
                    "iter: {iter:06d}",
                    "lr: {lr:.5f}",
                    '{meters}',
                    "eta: {eta}",
                    'mem: {mem}M',
                ]).format(
                    iter=iteration,
                    lr=optimizer.param_groups[0]['lr'],
                    meters=str(meters),
                    eta=eta_string,
                    mem=round(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0),
                )
            )
            if summary_writer:
                global_step = iteration
                summary_writer.add_scalar('losses/total_loss', loss, global_step=global_step)
                for loss_name, loss_item in loss_dict.items():
                    summary_writer.add_scalar('losses/{}'.format(loss_name), loss_item, global_step=global_step)
                summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        if iteration % cfg.SOLVER.SAVE_STEP == 0:
            checkpointer.save("model_{:06d}".format(iteration), **arguments)

        if cfg.SOLVER.EVAL_STEP > 0 and iteration % cfg.SOLVER.EVAL_STEP == 0 and not iteration == max_iter:
            eval_results = do_evaluation(cfg, model, iteration=iteration)
            for eval_result, dataset in zip(eval_results, cfg.DATASETS.TEST):
                write_metric(eval_result['metrics'], 'metrics/' + dataset, summary_writer, iteration)
            model.train()  # *IMPORTANT*: change to train mode after eval.

    checkpointer.save("model_final", **arguments)
    # compute training time
    total_training_time = int(time.time() - start_training_time)
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / max_iter))
    return model
