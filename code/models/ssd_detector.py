from torch import nn

from models.box_head.box_head import SSDBoxHead
from models.backbone_mobilenet import get_backbone_mobilenet_v2


class SSDDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # build_backbone(cfg)
        self.backbone = get_backbone_mobilenet_v2(cfg)
        self.box_head = SSDBoxHead(cfg)

    def forward(self, images, targets=None):
        features = self.backbone(images)
        detections, detector_losses = self.box_head(features, targets)
        if self.training:
            return detector_losses
        return detections
