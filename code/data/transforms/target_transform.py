import numpy as np
import torch

from utils import box_utils


class SSDTargetTransform:
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
        self.center_form_priors = center_form_priors
        # self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):

        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)

        if True or (gt_boxes.shape[-1]) == 3:
            gt_boxes = box_utils.corner_form_to_center_form(gt_boxes)
            gt_boxes[..., 2] += gt_boxes[..., 3]
            gt_boxes = torch.narrow(gt_boxes, 1, 0, 3)
            gt_boxes[..., 2] *= 0.5 * (4 / 3.14)    # scale up the circles to have same areas as squares
        # gt_boxes = box_utils.corner_form_to_center_form(gt_boxes)
        # gt_boxes[..., 2] += gt_boxes[..., 3]
        # gt_boxes = torch.narrow(gt_boxes, 2, 0, 3)
        # gt_boxes[..., 2] *= 0.5 * (4 / 3.14)    # scale up the circles to have same areas as squares

        boxes, labels = box_utils.assign_priors(gt_boxes, gt_labels,
                                                self.center_form_priors, self.iou_threshold)
        # boxes = box_utils.corner_form_to_center_form(boxes)
        locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance, self.size_variance)
       
        return locations, labels

