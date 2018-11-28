from __future__ import print_function
import caffe
import warnings
import numpy as np
import numpy.random as npr
import yaml
import easydict
from utils.cython_bbox import bbox_overlaps
import logging
import cv2
logger = logging.getLogger(__name__)

from utils.get_config import cfg
from utils.tensorboard import tb
from utils.bbox_transform import bbox_transform
from lib.layers.generate_anchors import generate_anchors


class MultiLayerAnchorLayer(caffe.Layer):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        self.iter = 0
        self.bottom = bottom
        self.top = top
        try:
            layer_params = yaml.load(self.param_str_)
        except AttributeError:
            layer_params = yaml.load(self.param_str)
        # Determine if hard negative mining should be performed
        # based on the number of bottom blobs
        self._source = layer_params['source']
        assert sorted(
            self._source) == self._source, 'source must be in ascending order'
        self._unique_source = list(set(self._source))
        self.level = [
            easydict.EasyDict({
                'idx':
                [_ for _ in range(len(self._source)) if self._source[_] == i]
            }) for i in range(len(self._unique_source))
        ]
        self._fg_fraction = layer_params.get(
            'fg_fraction', cfg.TRAIN.ANCHOR_SAMPLING.ANCHOR_FG_FRACTION)
        self._anchors_per_batch = layer_params.get(
            'anchors_per_batch', cfg.TRAIN.ANCHOR_SAMPLING.ANCHORS_PER_BATCH)
        self._shifts = layer_params.get('shifts', [0])
        self._hard_mining = layer_params.get('ohem', False)
        self._anchor_ratios = layer_params.get('ratios', (0.5, 1, 2))
        base_size = layer_params.get('base_size', 16)

        for i in self.level:
            i.source = self._source[i.idx[0]]
            i.anchor_scales = [layer_params['scales'][_] for _ in i.idx]
            i._feat_stride = [layer_params['feat_stride'][_] for _ in i.idx]
            i._anchors = generate_anchors(
                scales=np.array(i.anchor_scales),
                base_size=base_size,
                ratios=np.array(self._anchor_ratios),
                shifts=np.array(self._shifts),
                strides=np.array(i._feat_stride))
            i._num_anchors = i._anchors.shape[0]
            i._allowed_border = [
                layer_params['allowed_border'][_] for _ in i.idx
            ]
            i._positive_overlap = [
                layer_params.get(
                    'positive_overlap',
                    [cfg.TRAIN.ANCHOR_POSITIVE_OVERLAP] * len(self._source))[_]
                for _ in i.idx
            ]
            i.height, i.width = bottom[self._source[i.idx[0]]].data.shape[-2:]
            self._num_classes = bottom[self._source[
                i.idx[0]]].data.shape[1] / i._num_anchors

        for i, j in enumerate(self.level):
            # labels
            top[4 * i + 0].reshape(1, 1, j._num_anchors * j.height, j.width)
            # bbox_targets
            top[4 * i + 1].reshape(1, j._num_anchors * 4, j.height, j.width)
            # bbox_inside_weights
            top[4 * i + 2].reshape(1, j._num_anchors * 4, j.height, j.width)
            # bbox_outside_weights
            top[4 * i + 3].reshape(1, j._num_anchors * 4, j.height, j.width)

    def forward(self, bottom, top):
        bottom_data = [bottom[i].data for i in range(len(bottom))]
        assert bottom_data[0].shape[0] == 1, \
            'Only single item batches are supported'
        assert len(bottom_data) == len(self._unique_source) + 4, \
            'Num of input is incorrect, please check your prototxt'
        gt_boxes = bottom_data[len(self._unique_source)]
        # im_info and im_idx
        im_info = bottom_data[len(self._unique_source) + 1][0, :]
        im_idx = int(bottom_data[len(self._unique_source) + 3])

        self._skip_layer = np.zeros((len(self.level)), dtype=np.bool)

        # init level meta data
        for level_i, level in enumerate(self.level):
            assert len(set(level._feat_stride)) == 1, \
                'feature stride from same source must be same'
            assert len(set(level._allowed_border)) == 1, \
                'allowed border from same source must be same'
            assert len(set(level._positive_overlap)) == 1, \
                'positive overlap from same source must be same'

            source_idx = level.source
            # map of shape (..., H, W)
            level.height, level.width = bottom_data[source_idx].shape[-2:]
            # GT boxes (x1, y1, x2, y2, label)

            # 1. Generate proposals from bbox deltas and shifted anchors
            shift_x = np.arange(0, level.width) * level._feat_stride[0]
            shift_y = np.arange(0, level.height) * level._feat_stride[0]
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                                shift_x.ravel(), shift_y.ravel())).transpose()
            # add A anchors (1, A, 4) to
            # cell K shifts (K, 1, 4) to get
            # shift anchors (K, A, 4)
            # reshape to (K*A, 4) shifted anchors
            A = level._num_anchors
            K = shifts.shape[0]
            all_anchors = (level._anchors.reshape((1, A, 4)) + shifts.reshape(
                (1, K, 4)).transpose((1, 0, 2)))
            all_anchors = all_anchors.reshape((K * A, 4))
            level.total_anchors = int(K * A)

            # only keep anchors inside the image
            level.inds_inside = np.where(
                (all_anchors[:, 0] >= -level._allowed_border[0]) &
                (all_anchors[:, 1] >= -level._allowed_border[0]) &
                # width
                (all_anchors[:, 2] < im_info[1] + level._allowed_border[0]) &
                (all_anchors[:, 3] < im_info[0] + \
                 level._allowed_border[0])  # height
            )[0]

            # keep only inside anchors
            if level.inds_inside.size == 0:
                # If no anchors inside use whatever anchors we have
                level.inds_inside = np.arange(0, all_anchors.shape[0])

            level.all_anchors = all_anchors
            level.anchors = all_anchors[level.inds_inside, :]

            # label: 1 is positive, 0 is negative, -1 is don't care
            level.labels = np.empty((len(level.inds_inside), ),
                                    dtype=np.float32)
            level.labels.fill(-1)

            # overlaps between the anchors and the gt boxes
            # overlaps (ex, gt)

            level.overlaps = bbox_overlaps(
                np.ascontiguousarray(level.anchors, dtype=np.float),
                np.ascontiguousarray(gt_boxes, dtype=np.float))

        for level_i, level in enumerate(self.level):
            argmax_overlaps = level.overlaps.argmax(axis=1)
            max_overlaps = level.overlaps[np.arange(len(level.inds_inside)
                                                    ), argmax_overlaps]
            # assign bg labels first so that positive labels can clobber them
            level.labels[max_overlaps < cfg.TRAIN.ANCHOR_NEGATIVE_OVERLAP] = 0

            # fg label: above threshold IOU
            level.aboveIOUids = np.where(
                max_overlaps >= level._positive_overlap[0])[0]

            # Anchor level OHEM
            ohem_scores = bottom_data[level.source]
            ohem_scores_shape = ohem_scores.shape
            ohem_scores_shape_new = (ohem_scores_shape[0], self._num_classes,
                                     -1, ohem_scores_shape[-1])
            ohem_scores = ohem_scores.reshape(ohem_scores_shape_new)
            ohem_scores = _softmax(ohem_scores)
            ohem_scores_shape_new = (ohem_scores_shape[0], level._num_anchors,
                                     ohem_scores_shape[-2],
                                     ohem_scores_shape[-1])
            ohem_scores = ohem_scores[:, 0, ...].reshape(
                ohem_scores_shape_new).transpose((0, 2, 3, 1)).ravel()
            ohem_scores = ohem_scores[level.inds_inside]
            aligned_gts = argmax_overlaps[level.aboveIOUids]
            level.labels[level.aboveIOUids] = gt_boxes[aligned_gts, -1]
            # Subsample positives
            if self._fg_fraction >= 0:
                num_fg = int(self._fg_fraction * self._anchors_per_batch)
            else:
                num_fg = np.inf
            fg_inds_cls = np.where(
                max_overlaps >= level._positive_overlap[0])[0]
            # check for easy image
            pos_ohem_scores = 1 - ohem_scores[fg_inds_cls]
            if cfg.TRAIN.DISABLE_EASY_IMAGE.ENABLE and all(
                    pos_ohem_scores >= cfg.TRAIN.DISABLE_EASY_IMAGE.THRESHOLD):
                logger.debug(
                    '{} ignored at level {} by cfg.TRAIN.DISABLE_EASY_IMAGE, '
                    '{} faces recognized as easy, '
                    '{} faces in total, rank: {}, iter: {}'.format(
                        im_idx, level_i,
                        np.unique(argmax_overlaps[fg_inds_cls]).size,
                        gt_boxes.shape[0], cfg.RANK, self.iter))
                self._skip_layer[level_i] = True

            # we may not do mining at all if cfg.TRAIN.POSITIVE_MINING is off
            if len(fg_inds_cls) > num_fg and cfg.TRAIN.POSITIVE_MINING:
                if self._hard_mining:
                    # confidence for non-background
                    pos_ohem_scores = 1 - ohem_scores[fg_inds_cls]
                    order_pos_ohem_scores = pos_ohem_scores.argpartition(
                        num_fg)
                    ohem_sampled_fgs = fg_inds_cls[
                        order_pos_ohem_scores[num_fg:]]
                    level.labels[ohem_sampled_fgs] = -1
                else:
                    disable_inds = npr.choice(
                        range(len(fg_inds_cls[0])),
                        size=(len(fg_inds_cls[0]) - num_fg),
                        replace=False)
                    level.labels[fg_inds_cls[disable_inds]] = -1

            # Subsample negatives
            n_fg = np.sum(level.labels > 0)  # For multi-class
            if cfg.TRAIN.ANCHOR_SAMPLING.ANCHOR_NUM_METHOD == 'fixed_num':
                num_bg = self._anchors_per_batch - n_fg
            else:
                raise NotImplementedError('Unknown anchor num method')
            bg_inds = np.where(level.labels == 0)[0]
            if len(bg_inds) > num_bg:
                if not self._hard_mining:
                    # randomly sub-sample negatives
                    disable_inds = npr.choice(
                        range(len(bg_inds)),
                        size=(len(bg_inds) - num_bg),
                        replace=False)
                    level.labels[bg_inds[disable_inds]] = -1
                else:
                    # sort ohem scores
                    neg_ohem_scores = ohem_scores[bg_inds]
                    order_neg_ohem_scores = neg_ohem_scores.argpartition(
                        num_bg)
                    ohem_sampled_bgs = bg_inds[order_neg_ohem_scores[:num_bg]]
                    level.labels[bg_inds] = -1
                    level.labels[ohem_sampled_bgs] = 0

            fg_inds_reg = np.where(level.labels > 0)[0]

            if cfg.TRAIN.ANCHOR_REGRESSION_OVERLAP > 0:
                fg_inds_reg = np.where(
                    max_overlaps >= cfg.TRAIN.ANCHOR_REGRESSION_OVERLAP)[0]

            bbox_targets = _compute_targets(
                level.anchors[fg_inds_reg],
                gt_boxes[argmax_overlaps[fg_inds_reg], :])
            bbox_targets = _unmap(
                bbox_targets, level.inds_inside.size, fg_inds_reg, fill=-1)
            bbox_inside_weights = np.zeros((len(level.inds_inside), 4),
                                           dtype=np.float32)
            bbox_inside_weights[fg_inds_reg, :] = np.array(
                cfg.TRAIN.BBOX_INSIDE_WEIGHTS)  # For multi-class
            bbox_outside_weights = np.zeros((len(level.inds_inside), 4),
                                            dtype=np.float32)

            # uniform weighting of examples (given non-uniform sampling)
            num_examples = fg_inds_reg.size
            if num_examples > 0:
                positive_weights = np.ones((1, 4)) * 1.0 / num_examples
            else:
                positive_weights = np.zeros((1, 4))
            bbox_outside_weights[fg_inds_reg] = positive_weights
            try:
                level.accuracy = 1.0 * (
                    np.sum(ohem_scores[level.labels > 0] <
                           (1.0 - cfg.MISC.ACCURACY_THRESHOLD)) +
                    np.sum(ohem_scores[level.labels == 0] >= cfg.MISC.
                           ACCURACY_THRESHOLD)) / np.sum(level.labels >= 0)
            except:
                level.accuracy = 1.0
            labels = _unmap(
                level.labels, level.total_anchors, level.inds_inside, fill=-1)
            bbox_targets = _unmap(
                bbox_targets, level.total_anchors, level.inds_inside, fill=0)
            bbox_inside_weights = _unmap(
                bbox_inside_weights,
                level.total_anchors,
                level.inds_inside,
                fill=0)
            bbox_outside_weights = _unmap(
                bbox_outside_weights,
                level.total_anchors,
                level.inds_inside,
                fill=0)

            # labels TODO: why should we transpose anchor dim from 3 to 1 and
            #              then back to 3?
            labels = labels.reshape((1, level.height, level.width,
                                     level._num_anchors)).transpose(
                                         0, 3, 1, 2)
            labels = labels.reshape((1, 1, level._num_anchors * level.height,
                                     level.width))
            top[level_i * 4 + 0].reshape(*labels.shape)
            top[level_i * 4 + 0].data[...] = labels

            # bbox_targets
            bbox_targets = bbox_targets \
                .reshape((1, level.height, level.width, level._num_anchors * 4)).transpose(0, 3, 1, 2)
            top[level_i * 4 + 1].reshape(*bbox_targets.shape)
            top[level_i * 4 + 1].data[...] = bbox_targets

            # bbox_inside_weights
            bbox_inside_weights = bbox_inside_weights \
                .reshape((1, level.height, level.width, level._num_anchors * 4)).transpose(0, 3, 1, 2)
            assert bbox_inside_weights.shape[2] == level.height
            assert bbox_inside_weights.shape[3] == level.width
            top[level_i * 4 + 2].reshape(*bbox_inside_weights.shape)
            top[level_i * 4 + 2].data[...] = bbox_inside_weights

            # bbox_outside_weights
            bbox_outside_weights = bbox_outside_weights \
                .reshape((1, level.height, level.width, level._num_anchors * 4)).transpose(0, 3, 1, 2)
            assert bbox_outside_weights.shape[2] == level.height
            assert bbox_outside_weights.shape[3] == level.width
            top[level_i * 4 + 3].reshape(*bbox_outside_weights.shape)
            top[level_i * 4 + 3].data[...] = bbox_outside_weights
        self.accuracy = np.mean([i.accuracy for i in self.level])
        if cfg.TRAIN.DISABLE_EASY_IMAGE.ENABLE:
            if np.all(self._skip_layer):
                if not cfg.TRAIN.DISABLE_EASY_IMAGE.SMOOTH and npr.rand(
                ) <= cfg.TRAIN.DISABLE_EASY_IMAGE.PROB:
                    self._roidb[im_idx]['skip'] = np.inf
                    logger.warning('{} ignored at rank: {}'.format(
                        im_idx, cfg.RANK))
                if cfg.TRAIN.DISABLE_EASY_IMAGE.SMOOTH:
                    self._roidb[im_idx][
                        'prob'] = cfg.TRAIN.DISABLE_EASY_IMAGE.PROB

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def set_roidb(self, roidb):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb

    def set_iter(self, iter):
        """Set the iter number in this object."""
        self.iter = iter


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(
        np.float32, copy=False)


def _softmax(x):
    e_x = np.exp(x - np.max(x, axis=1)[np.newaxis, ...].transpose((1, 0, 2,
                                                                   3)))
    return e_x / e_x.sum(axis=1)[np.newaxis, ...].transpose((1, 0, 2, 3))
