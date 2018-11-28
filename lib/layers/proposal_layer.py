import caffe
import numpy as np
import yaml

from utils.get_config import cfg
from utils.bbox_transform import bbox_transform_inv, clip_boxes
from lib.layers.generate_anchors import generate_anchors
from nms.nms_wrapper import nms

DEBUG = False


class ProposalLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        try:
            layer_params = yaml.load(self.param_str_)
        except AttributeError:
            layer_params = yaml.load(self.param_str)

        self._feat_stride = layer_params['feat_stride']
        self._shifts = layer_params.get('shifts', [0])
        self._anchor_ratios = layer_params.get('ratios', (0.5, 1, 2))
        anchor_scales = layer_params.get('scales', (8, 16, 32))
        base_size = layer_params.get('base_size', 16)
        self._anchors = generate_anchors(
            scales=np.array(anchor_scales),
            base_size=base_size,
            ratios=np.array(self._anchor_ratios),
            shifts=np.array(self._shifts),
            strides=np.array(self._feat_stride))
        self._num_anchors = self._anchors.shape[0]

        self._subsampled = layer_params.get('subsampled', True)

        self._num_feats = layer_params.get('num_feats', 1)

        self._refine = (len(bottom) > 3)
        self._num_refine = len(bottom) - 3

        if DEBUG:
            print 'feat_stride: {}'.format(self._feat_stride)
            print 'anchors:'
            print self._anchors

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[0].reshape(1, 5)

        # scores blob: holds scores for R regions of interest
        if len(top) > 1:
            top[1].reshape(1, 1, 1, 1)

    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)

        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'
        if self.phase == 0:
            cfg_key = 'TRAIN'
        elif self.phase == 1:
            cfg_key = 'TEST'
        else:
            cfg_key = str(self.phase)  # either 'TRAIN' or 'TEST'

        if cfg_key == 'TRAIN':
            nms_thresh = cfg[cfg_key].NMS_THRESH
            post_nms_topN = cfg[cfg_key].ANCHOR_N_POST_NMS
            pre_nms_topN = cfg[cfg_key].ANCHOR_N_PRE_NMS

        if cfg_key == 'TEST':
            pre_nms_topN = cfg[cfg_key].N_DETS_PER_MODULE
            score_thresh = cfg[cfg_key].SCORE_THRESH

        min_size = cfg[cfg_key].ANCHOR_MIN_SIZE

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        scores = bottom[-3].data  # For multi-class
        bbox_deltas = bottom[-2].data
        im_info = bottom[-1].data[0, :]

        # 1. Generate proposals from bbox deltas and shifted anchors
        height, width = scores.shape[-2:]

        # Enumerate all shifts
        shift_x = np.arange(0, width) * self._feat_stride[0]
        shift_y = np.arange(0, height) * self._feat_stride[0]
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(),
                            shift_y.ravel())).transpose()

        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        num_classes = scores.shape[1] / (A * self._num_feats)
        anchors = self._anchors.reshape((1, A, 4)) + \
            shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4))
        self.anchors = anchors

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #
        # bbox deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

        # Same story for the scores:
        #
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        scores = scores.transpose((0, 2, 3, 1)).reshape(
            (-1, num_classes, A * self._num_feats)).transpose(
                (0, 2, 1)).reshape((-1, num_classes))

        # Convert anchors into proposals via bbox transformations
        new_anchors = np.concatenate(
            [anchors[:, np.newaxis, :]] * self._num_feats, axis=1).reshape((-1,
                                                                            4))
        proposals = bbox_transform_inv(new_anchors, bbox_deltas)
        for i in range(self._num_refine):
            # Do this because a combination of bbox_transform_inv and _compute_targets
            # will cause a larger 3rd and 4th entry of coordinates
            # We do not do this at the last regression, just to follow the original code
            proposals[:, 2:4] -= 1
            refine_delta = bottom[i].data
            refine_delta = refine_delta.transpose((0, 2, 3, 1)).reshape(
                (-1, 4))
            proposals = bbox_transform_inv(proposals, refine_delta)

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info[:2])

        if self._subsampled:
            anchor_map = np.zeros((height, width, A))
            for i in xrange(A):
                stride = self._feat_stride[i / len(self._shifts)**
                                           2] // self._feat_stride[0]
                anchor_map[::stride, ::stride, i] = 1
            anchor_map = anchor_map.reshape((K * A))
            subsampled_inds = np.where(anchor_map)[0]
            proposals = proposals[subsampled_inds, :]
            scores = scores[subsampled_inds, :]

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = _filter_boxes(proposals, min_size * im_info[2])
        proposals = proposals[keep, :]
        scores = scores[keep, :]

        # # 4. sort all (proposal, score) pairs by score from highest to lowest
        # # 5. take top pre_nms_topN
        #
        max_score = np.max(scores[:, 1:], axis=1).ravel()
        order = max_score.argsort()[::-1]
        try:
            thresh_idx = np.where(max_score[order] >= score_thresh)[0].max()
        except:
            thresh_idx = 0  # Nothing greater then score_thresh, just keep the largest one
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        order = order[:thresh_idx + 1]
        proposals = proposals[order, :]
        scores = scores[order, :]

        # 6. apply nms (if in training mode)
        # 7. take after_nms_topN
        # 8. return the top proposals (-> RoIs top)
        if self.phase == 0:
            # DO NMS ONLY IN TRAINING TIME
            # DURING TEST WE HAVE NMS OUTSIDE OF THIS FUNCTION
            keep = nms(np.hstack((proposals, scores)), nms_thresh)
            if post_nms_topN > 0:
                keep = keep[:post_nms_topN]
            proposals = proposals[keep, :]
            scores = scores[keep]

        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        if proposals.shape[0] == 0:
            blob = np.array([[0, 0, 0, 16, 16]], dtype=np.float32)
        else:
            batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
            blob = np.hstack((batch_inds,
                              proposals.astype(np.float32, copy=False)))

        top[0].reshape(*(blob.shape))
        top[0].data[...] = blob

        # [Optional] output scores blob
        if len(top) > 1:
            top[1].reshape(*(scores.shape))
            top[1].data[...] = scores

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep
