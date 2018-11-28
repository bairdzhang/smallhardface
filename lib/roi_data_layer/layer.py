'''
Data fetching layer for training
'''
import caffe
import numpy as np
import yaml
import logging
logger = logging.getLogger(__name__)

import numpy.random as npr
from utils.get_config import cfg
from roi_data_layer.minibatch import get_minibatch


class RoIDataLayer(caffe.Layer):
    def _shuffle_roidb_inds(self, rank=0):
        self.rank = rank
        """Randomly permute the training roidb."""
        if cfg.TRAIN.ASPECT_GROUPING:
            widths = np.array([r['width'] for r in self._original_roidb])
            heights = np.array([r['height'] for r in self._original_roidb])
            horz = (widths >= heights)
            vert = np.logical_not(horz)
            horz_inds = np.where(horz)[0]
            vert_inds = np.where(vert)[0]
            inds = np.hstack((np.random.permutation(horz_inds),
                              np.random.permutation(vert_inds)))
            inds = np.reshape(inds, (-1, 2))
            np.random.seed(rank)
            row_perm = np.random.permutation(np.arange(inds.shape[0]))
            inds = np.reshape(inds[row_perm, :], (-1, ))
            self._perm = inds
        else:
            self._perm = np.random.permutation(
                np.arange(len(self._original_roidb)))
        skiped_images = []
        if cfg.TRAIN.DISABLE_EASY_IMAGE.SMOOTH:
            for i, j in enumerate(self._original_roidb):
                if 'prob' in j and npr.rand() < j['prob']:
                    skiped_images.append(i)
        else:
            for i, j in enumerate(self._original_roidb):
                if 'skip' in j and j['skip'] >= 1:
                    skiped_images.append(i)
                    j['skip'] -= 1
        if len(skiped_images) > 0:
            self._perm = [i for i in self._perm if i not in skiped_images]
            logger.warning(
                '{} images disabled, {} images left, rank: {}'.format(
                    len(skiped_images), len(self._perm), cfg.RANK))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + cfg.TRAIN.IMS_PER_BATCH > len(self._perm):
            logger.info('New epoch, rank: {}'.format(self.rank))
            self._shuffle_roidb_inds(self.rank)

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def _get_next_minibatch(self, scale_idx):
        """Return the blobs to be used for the next minibatch.
        """

        def merge_two_dicts(x, y):
            """Given two dicts, merge them into a new dict as a shallow copy."""
            z = x.copy()
            z.update(y)
            return z

        self.db_inds = self._get_next_minibatch_inds()
        minibatch_db = [
            merge_two_dicts(self._roidb[i], {"idx": i})
            for i in self.db_inds
        ]
        res, self.resize_ratio = get_minibatch(
            minibatch_db,
            scale_idx)
        for idx, scale_i in zip(self.db_inds, self.resize_ratio):
            if 'seen_scale' not in self._roidb[idx]:
                self._roidb[idx]['seen_scale'] = []
            self._roidb[idx]['seen_scale'].append(scale_i)
            logger.debug('Scale {} seeing for {}, rank: {}'.format(
                scale_i, idx, cfg.RANK))
        return res

    def set_roidb(self, roidb, rank=0):
        """Set the roidb to be used by this layer during training."""
        self._original_roidb = roidb
        self._roidb = roidb
        self.flag = np.ones((len(self._roidb)), dtype=np.bool)
        self._shuffle_roidb_inds(rank)

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""
        self.bottom = bottom
        self.top = top

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)

        self._num_classes = layer_params['num_classes']

        self._name_to_top_map = {}

        # data blob: holds a batch of N images, each with 3 channels
        idx = 0
        top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, 3, cfg.TRAIN.SCALES.MAX_SIZE,
                         cfg.TRAIN.SCALES.MAX_SIZE)
        self._name_to_top_map['data'] = idx
        idx += 1

        top[idx].reshape(1, 3)
        self._name_to_top_map['im_info'] = idx
        idx += 1

        top[idx].reshape(1, 4)
        self._name_to_top_map['gt_boxes'] = idx
        idx += 1

        top[idx].reshape(1)
        self._name_to_top_map['im_idx'] = idx
        idx += 1

        self.epoch_ind = 0

        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top, scale_idx=-1):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch(scale_idx)

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
