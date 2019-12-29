import torch
from torch import nn
import torch.nn.functional as F

from utils import box_utils
from .anchors import PriorBox
from .box_predictor import SSDBoxPredictor
from .inference import PostProcessor
from .loss import MultiBoxLoss


class SSDBoxHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.predictor = SSDBoxPredictor(cfg)
        self.loss_evaluator = MultiBoxLoss(neg_pos_ratio=cfg.MODEL.NEG_POS_RATIO)
        self.post_processor = PostProcessor(cfg)
        self.priors = None

    def forward(self, features, targets=None):
        cls_logits, bbox_pred = self.predictor(features)
        if self.training:
            return self._forward_train(cls_logits, bbox_pred, targets)
        else:
            return self._forward_test(cls_logits, bbox_pred)

    def _forward_train(self, cls_logits, bbox_pred, targets):
        # drop the height of boxes
        gt_boxes, gt_labels = targets['boxes'], targets['labels']

        # average box w and h into side length
        # for i, boxes_batch in enumerate(gt_boxes):
        #     for j, box in enumerate(boxes_batch):
        #         # drop last item
        #         box[2] = (box[2] + box[3]) / 2
        # print(gt_boxes[:][:][:][2])
        # print(gt_boxes[:][:][:][2] + gt_boxes[:][:][:][3])
        # gt_boxes = torch.narrow(gt_boxes, 2, 0, 3)

        reg_loss, cls_loss = self.loss_evaluator(cls_logits, bbox_pred, gt_labels, gt_boxes)
        loss_dict = dict(
            reg_loss=reg_loss,
            cls_loss=cls_loss,
        )
        detections = (cls_logits, bbox_pred)
        return detections, loss_dict

    def _forward_test(self, cls_logits, bbox_pred):
        if self.priors is None:
            self.priors = PriorBox(self.cfg)().to(bbox_pred.device)
        scores = F.softmax(cls_logits, dim=2)
        #print(bbox_pred[0])
        #print(self.priors[0])
        boxes = box_utils.convert_locations_to_boxes(
            bbox_pred, self.priors, self.cfg.MODEL.CENTER_VARIANCE, self.cfg.MODEL.SIZE_VARIANCE
        )
        boxes = box_utils.center_form_to_corner_form(boxes)
        detections = (scores, boxes)
        detections = self.post_processor(detections)
        # for box in detections[0]['boxes']:
        #     print(box[2] - box[0])
        #     print(box[3] - box[1])
        #     print('-----')
        return detections, {}
