import torch
import torch.nn as nn
import torch.nn.functional as F


import cv2
import random
import numpy as np
from torch import nn
from mmdet.models.builder import build_head
from mmocr.models.builder import DETECTORS
from mmocr.models.textdet.detectors import SingleStageTextDetector



@DETECTORS.register_module()
class BridgeSpotter(SingleStageTextDetector):
    def __init__(
            self
    ):
        super(Bridge, self).__init__()