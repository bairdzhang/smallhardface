# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from utils.get_config import cfg
from nms.cpu_nms import cpu_nms
# from nms.gpu_nms import gpu_nms
import numpy as np

def nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    return cpu_nms(dets, thresh)
 #   if cfg.USE_GPU_NMS and not force_cpu:
 #       return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
 #   else:
 #       return cpu_nms(dets, thresh)
