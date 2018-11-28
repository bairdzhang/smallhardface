from __future__ import print_function
import caffe
import numpy as np
import numpy.random as npr
import yaml
import easydict
from utils.cython_bbox import bbox_overlaps
import logging
logger = logging.getLogger(__name__)

from utils.get_config import cfg
from utils.bbox_transform import bbox_transform, bbox_transform_inv
from lib.layers.generate_anchors import generate_anchors


class MergePrediction(caffe.Layer):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        self.bottom = bottom
        self.top = top
        try:
            layer_params = yaml.load(self.param_str_)
        except AttributeError:
            layer_params = yaml.load(self.param_str)
        top[0].reshape(*bottom[0].shape)

    def forward(self, bottom, top):
        bottom_data = np.concatenate(
            [bottom[i].data for i in range(len(bottom))], axis=0)
        n, c, h, w = bottom_data.shape
        sftmx = _softmax(bottom_data.reshape((n, 2, -1, w)))[:, 0, ...]
        # strongest means smallest background confidence
        strongest = np.min(sftmx, axis=0)
        top[0].reshape(*bottom[0].shape)
        top[0].data[:] = np.stack((strongest, 1-strongest)).reshape(bottom[0].shape)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _softmax(x):
    e_x = np.exp(x - np.max(x, axis=1)[np.newaxis, ...].transpose((1, 0, 2,
                                                                   3)))
    return e_x / e_x.sum(axis=1)[np.newaxis, ...].transpose((1, 0, 2, 3))
