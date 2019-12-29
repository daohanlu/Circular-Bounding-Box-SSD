import warnings
import torchvision
import torch

try:
    import torch_extension

    _nms = torch_extension.nms
except ImportError:
    if torchvision.__version__ >= '0.3.0':
        _nms = torchvision.ops.nms
    else:
        from .python_nms import python_nms

        _nms = python_nms
        warnings.warn('You are using python version NMS because PyTorch version < 0.3.0')


def boxes_nms(boxes, scores, nms_thresh, max_count=-1):
    """ Performs non-maximum suppression, run on GPU or CPU according to
    boxes's device.
    Args:
        boxes(Tensor): `xyxy` mode boxes, use absolute coordinates(or relative coordinates), shape is (n, 4)
        scores(Tensor): scores, shape is (n, )
        nms_thresh(float): thresh
        max_count (int): if > 0, then only the top max_proposals are kept  after non-maximum suppression
    Returns:
        indices kept.
    """
    # for box in boxes:
    #     # duplicate the side length of the box for nms
    #     box = torch.cat((box.view(1), box[-1]))
    keep = _nms(boxes, scores, nms_thresh)
    if max_count > 0:
        keep = keep[:max_count]
    return keep
