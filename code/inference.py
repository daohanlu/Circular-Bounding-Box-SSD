import logging
import os
import random

import torch
import torch.utils.data
from tqdm import tqdm

from data.loader import make_data_loader
from data.datasets.evaluation import evaluate

from utils import mkdir


def compute_on_dataset(model, data_loader, device, sample_rate=1):
    results_dict = {}
    if sample_rate < 1 and sample_rate > 0:
        # data_loader.dataset = torch.utils.data.random_split(data_loader.dataset, len(data_loader)*sample_rate)[0]
        pass
    elif sample_rate != 1:
        raise AttributeError('sample_rate must be >0 and <= 1!')
    for batch in tqdm(data_loader):
        images, targets, image_ids = batch
        cpu_device = torch.device("cpu")
        with torch.no_grad():
            outputs = model(images.to(device))
            outputs = [o.to(cpu_device) for o in outputs]
        results_dict.update(
            {img_id.item(): result for img_id, result in zip(image_ids, outputs)}
        )
    return results_dict


def inference(model, data_loader, dataset_name, device, sample_rate=1, output_folder=None, use_cached=False, **kwargs):
    dataset = data_loader.dataset
    logger = logging.getLogger("SSD.inference")
    logger.info("Evaluating {} dataset({} images):".format(dataset_name, len(dataset)))
    predictions_path = os.path.join(output_folder, 'predictions.pth')
    if use_cached and os.path.exists(predictions_path):
        predictions = torch.load(predictions_path, map_location='cpu')
    else:
        predictions = compute_on_dataset(model, data_loader, device, sample_rate=sample_rate)
    if output_folder:
        torch.save(predictions, predictions_path)
    return evaluate(dataset=dataset, predictions=predictions, output_dir=output_folder, **kwargs)


@torch.no_grad()
def do_evaluation(cfg, model, sample_rate=1, **kwargs):
    model.eval()
    device = torch.device(cfg.MODEL.DEVICE)
    data_loaders_val = make_data_loader(cfg, is_train=False)
    eval_results = []
    for dataset_name, data_loader in zip(cfg.DATASETS.TEST, data_loaders_val):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        if not os.path.exists(output_folder):
            mkdir(output_folder)
        eval_result = inference(model, data_loader, dataset_name, device, sample_rate, output_folder, **kwargs)
        eval_results.append(eval_result)
    return eval_results
