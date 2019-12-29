from itertools import product

import torch
from math import sqrt


class PriorBox:
    def __init__(self, cfg):
        self.image_size = cfg.INPUT.IMAGE_SIZE
        prior_config = cfg.MODEL.PRIORS
        self.feature_maps = prior_config.FEATURE_MAPS
        self.min_sizes = prior_config.MIN_SIZES
        self.max_sizes = prior_config.MAX_SIZES
        self.strides = prior_config.STRIDES
        self.additional_zooms = prior_config.ADDITIONAL_ZOOMS
        self.clip = prior_config.CLIP

    def __call__(self):
        """Generate SSD Prior Boxes.
            It returns the center, height and width of the priors. The values are relative to the image size
            Returns:
                priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
                    are relative to the image size.
        """
        priors = []
        for k, f in enumerate(self.feature_maps):
            scale = self.image_size / self.strides[k]
            for i, j in product(range(f), repeat=2):
                # unit center x,y
                cx = (j + 0.5) / scale
                cy = (i + 0.5) / scale

                # small sized square box
                size = self.min_sizes[k]
                h = w = size / self.image_size
                priors.append([cx, cy, w])

                # # big sized square box
                # size = sqrt(self.min_sizes[k] * self.max_sizes[k])
                # h = w = size / self.image_size
                # priors.append([cx, cy, w, w])
                #
                # # change h/w ratio of the small sized box
                # min_size = self.min_sizes[k]
                # max_size = sqrt(self.min_sizes[k] * self.max_sizes[k])
                # min_h = min_w = min_size / self.image_size
                # max_h = max_w = max_size / self.image_size
                # for zoom in self.additional_zooms:
                #     # side_length(zoom) means side_length with zoom as a percentage from min size to max size
                #     side_length = min_size + zoom * (max_h - min_h)
                #     priors.append([cx, cy, side_length, side_length])
                # print('num priors:' + str(len(priors)))

        priors = torch.tensor(priors)
        if self.clip:
            priors.clamp_(max=1, min=0)
        return priors
