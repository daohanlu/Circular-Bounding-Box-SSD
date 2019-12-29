import glob
import os
import time

import torch
from PIL import Image
from utils.draw import draw_boxes

import argparse
import numpy as np

from data.datasets import COCODataset, VOCDataset
from data.transforms import build_transforms
from models.ssd_detector import SSDDetector
from utils import mkdir
from utils.checkpoint import CheckPointer
from config import get_default_config


@torch.no_grad()
def run_demo(cfg, ckpt, score_threshold, images_dir, output_dir, dataset_type):
    if dataset_type == "voc":
        class_names = VOCDataset.class_names
    elif dataset_type == 'coco':
        class_names = COCODataset.class_names
    else:
        raise NotImplementedError('Not implemented now.')
    device = torch.device(cfg.MODEL.DEVICE)

    model = SSDDetector(cfg)
    model = model.to(device)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print('Loaded weights from {}'.format(weight_file))

    image_paths = glob.glob(os.path.join(images_dir, '*.jpg'))
    mkdir(output_dir)

    cpu_device = torch.device("cpu")
    transforms = build_transforms(cfg, is_train=False)
    model.eval()
    for i, image_path in enumerate(image_paths):
        start = time.time()
        image_name = os.path.basename(image_path)

        image = np.array(Image.open(image_path).convert("RGB"))
        height, width = image.shape[:2]
        images = transforms(image)[0].unsqueeze(0)
        load_time = time.time() - start

        start = time.time()
        result = model(images.to(device))[0]
        inference_time = time.time() - start

        result = result.resize((width, height)).to(cpu_device).numpy()
        boxes, labels, scores = result['boxes'], result['labels'], result['scores']
        # print("[demo.py] file: {} boxes: {}".format(image_path, boxes))
        # get minimum square box
        # if height < width:
        #     boxes[:, 2] = boxes[:, 0] + (boxes[:, 3] - boxes[:, 1])
        # else:
        #     boxes[:, 3] = boxes[:, 1] + (boxes[:, 2] - boxes[:, 0])

        indices = scores > score_threshold
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]
        meters = ' | '.join(
            [
                'objects {:02d}'.format(len(boxes)),
                'load {:03d}ms'.format(round(load_time * 1000)),
                'inference {:03d}ms'.format(round(inference_time * 1000)),
                'FPS {}'.format(round(1.0 / inference_time))
            ]
        )
        print('({:04d}/{:04d}) {}: {}'.format(i + 1, len(image_paths), image_name, meters))

        drawn_image = draw_boxes(image, boxes, labels, scores, class_names).astype(np.uint8)
        pil_img = Image.fromarray(drawn_image)
        pil_img = pil_img.resize((max(pil_img.size), max(pil_img.size)))
        pil_img.save(os.path.join(output_dir, image_name))


def main():
    cfg = get_default_config()
    parser = argparse.ArgumentParser(description="SSD Demo.")
    parser.add_argument("--ckpt", type=str, default=None, help="Trained weights.")
    parser.add_argument("--score_threshold", type=float, default=0.1)
    parser.add_argument("--images_dir", default='demo', type=str, help='Specify a image dir to do prediction.')
    parser.add_argument("--output_dir", default='demo/result', type=str, help='Specify a image dir to save predicted images.')
    parser.add_argument("--dataset_type", default="voc", type=str, help='Specify dataset type. Currently support voc and coco.')

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    print(args)

    cfg.merge_from_list(args.opts)
    cfg.freeze()

    print("Running with config:\n{}".format(cfg))

    run_demo(cfg=cfg,
             ckpt=args.ckpt,
             score_threshold=args.score_threshold,
             images_dir=args.images_dir,
             output_dir=args.output_dir,
             dataset_type=args.dataset_type)


if __name__ == '__main__':
    main()