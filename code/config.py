from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.META_ARCHITECTURE = 'SSDDetector'
_C.MODEL.DEVICE = "cuda"
# match default boxes to any ground truth with jaccard overlap higher than a threshold (0.5)
_C.MODEL.THRESHOLD = 0.5
_C.MODEL.NUM_CLASSES = 21
# Hard negative mining
_C.MODEL.NEG_POS_RATIO = 3
_C.MODEL.CENTER_VARIANCE = 0.1
_C.MODEL.SIZE_VARIANCE = 0.2

# -----------------------------------------------------------------------------
# PRIORS
# -----------------------------------------------------------------------------
_C.MODEL.PRIORS = CN()
_C.MODEL.PRIORS.FEATURE_MAPS = [20, 10, 5, 3, 2, 1]
_C.MODEL.PRIORS.STRIDES = [16, 32, 64, 100, 150, 300]
_C.MODEL.PRIORS.MIN_SIZES = [60, 105, 150, 195, 240, 285]
_C.MODEL.PRIORS.MAX_SIZES = [105, 150, 195, 240, 285, 330]
# _C.MODEL.PRIORS.ASPECT_RATIOS = [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
# instead of having aspect ratios, the circular anchors will have additional anchors
# a zoom of 0 means same size as min size; zoom of 1 means same size as max size
_C.MODEL.PRIORS.ADDITIONAL_ZOOMS = [0.3, 0.7]
#_C.MODEL.PRIORS.BOXES_PER_LOCATION = [4, 4, 4, 4, 4, 4]
_C.MODEL.PRIORS.BOXES_PER_LOCATION = [1, 1, 1, 1, 1, 1]
_C.MODEL.PRIORS.CLIP = True
    # backbone / output
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.OUT_CHANNELS = (96, 1280, 512, 256, 256, 64)

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Image size
_C.INPUT.IMAGE_SIZE = 320
# Values to be used for image normalization, RGB layout
_C.INPUT.PIXEL_MEAN = [123, 117, 104]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ("voc_2012_trainval",)
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ("voc_2007_test",)

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATA_LOADER = CN()
# Number of data loading threads
_C.DATA_LOADER.NUM_WORKERS = 8
_C.DATA_LOADER.PIN_MEMORY = True

# ---------------------------------------------------------------------------- #
# Solver: Training Parameters
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# tSOLVERfigs
_C.SOLVER.MAX_ITER = 120000
# sSOLVERn momentum
_C.SOLVER.LR_STEPS = [80000, 100000]
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.BATCH_SIZE = 32
_C.SOLVER.LR = 1e-3
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 5e-4
_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500

_C.SOLVER.LOG_STEP = 100
_C.SOLVER.SAVE_STEP = 2000
_C.SOLVER.EVAL_STEP = 4000

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.NMS_THRESHOLD = 0.45
_C.TEST.CONFIDENCE_THRESHOLD = 0.01
_C.TEST.MAX_PER_CLASS = -1
_C.TEST.MAX_PER_IMAGE = 100
_C.TEST.BATCH_SIZE = 10

_C.OUTPUT_DIR = 'outputs/ssd_mobilnet_v2'


def get_default_config():
    return _C
