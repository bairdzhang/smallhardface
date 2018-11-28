import cv2
import numpy as np
import numpy.random as npr
from utils.get_config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob
from random import choice
from utils.cython_bbox import bbox_overlaps
import logging
logger = logging.getLogger(__name__)


def get_minibatch(minibatch_db,
                  scale_idx):
    """Return the mini-batch for training"""
    num_images = len(minibatch_db)
    if cfg.TRAIN.SCALES.MODE == "SHORT_SIDE":
        num_scales = len(cfg.TRAIN.SCALES.SHORT_SIDE)
    else:
        raise NotImplementedError("Unknown TRAIN.SCALES.MODE: {}".format(
            cfg.TRAIN.SCALES.MODE))
    if scale_idx == -1:
        random_scale_inds = npr.randint(
            0, high=num_scales, size=num_images)
    else:
        random_scale_inds = np.array([scale_idx])  # DEBUG only

    im_blob, im_scales = _get_image_blob(minibatch_db, random_scale_inds)

    blobs = {'data': im_blob}

    assert len(im_scales) == 1, "Single batch only"
    assert len(minibatch_db) == 1, "Single batch only"
    # gt boxes: (x1, y1, x2, y2, cls)
    if ('in_memory' in minibatch_db[0] and minibatch_db[0]['in_memory']):
        blobs['gt_boxes'] = minibatch_db[0]['bbox'].copy()
    else:
        gt_inds = np.where(minibatch_db[0]['gt_classes'] != 0)[0]
        gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
        gt_boxes[:, 0:4] = minibatch_db[0]['boxes'][gt_inds, :] * im_scales[0]
        gt_boxes[:, 4] = minibatch_db[0]['gt_classes'][gt_inds]
        blobs['gt_boxes'] = gt_boxes
    if (cfg.TRAIN.AUGMENT.ENABLE and npr.rand() < cfg.TRAIN.AUGMENT.CROP.PROB):
        blobs = _crop_blobs(blobs)
    blobs['im_info'] = np.array(
        [[blobs['data'].shape[2], blobs['data'].shape[3]]], dtype=np.float32)
    blobs['im_idx'] = minibatch_db[0]['idx']
    # We pad the blobs['data'] to satisfy MAX_RESOLUTION
    # Note that we do not change gt_boxes or im_info
    h, w = blobs['data'].shape[2:]
    new_h = int(np.ceil(1.0 * h / cfg.MAX_RESOLUTION) * cfg.MAX_RESOLUTION)
    new_w = int(np.ceil(1.0 * w / cfg.MAX_RESOLUTION) * cfg.MAX_RESOLUTION)
    blobs['data'] = np.pad(blobs['data'], ((0, 0), (0, 0), (0, new_h - h),
                                           (0, new_w - w)), 'constant')
    return blobs, random_scale_inds


def _get_image_blob(roidb, scale_inds):
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in xrange(num_images):
        if ('in_memory' in roidb[i] and roidb[i]['in_memory']):
            im = roidb[i]['img'].copy()
            target_size = -1
            logger.debug('found image with in_memory tag')
        else:
            im = cv2.imread(roidb[i]['image'])

            if roidb[i]['flipped']:
                im = im[:, ::-1, :]
            face_median = np.median(
                (roidb[i]['boxes'][:, 2] - roidb[i]['boxes'][:, 0]) *
                (roidb[i]['boxes'][:, 3] - roidb[i]['boxes'][:, 1]))
            if np.isnan(face_median) or face_median == 0:
                valid_face_median = False
            else:
                valid_face_median = True

            if cfg.TRAIN.SCALES.MODE == "SHORT_SIDE":
                mode = "SHORT_SIDE"
                target_size = cfg.TRAIN.SCALES.SHORT_SIDE[scale_inds[i]]
            else:
                raise NotImplementedError(
                    'Unknown TRAIN.SCALES.MODE: {}'.format(
                        cfg.TRAIN.SCALES.MODE))
        im, im_scale = prep_im_for_blob(
            im,
            np.array(cfg.PIXEL_MEANS),
            target_size,
            cfg.TRAIN.SCALES.MAX_SIZE,
            mode=mode,
            face_median=face_median)
        im_scales.append(im_scale)
        processed_ims.append(im)

    blob = im_list_to_blob(processed_ims)
    return blob, im_scales


def _crop_blobs(blobs):
    img_height, img_width = blobs['data'].shape[2:]
    flag = False
    for i in range(cfg.TRAIN.AUGMENT.CROP.MAX_TRIES):
        width_ratio = npr.uniform(cfg.TRAIN.AUGMENT.CROP.LOWER,
                                  cfg.TRAIN.AUGMENT.CROP.UPPER)
        height_ratio = npr.uniform(cfg.TRAIN.AUGMENT.CROP.LOWER,
                                   cfg.TRAIN.AUGMENT.CROP.UPPER)
        height = int(
            np.clip(np.round(img_height * height_ratio), 0, img_height))
        width = int(np.clip(np.round(img_width * width_ratio), 0, img_width))
        height_start = npr.randint(img_height - height + 1)
        width_start = npr.randint(img_width - width + 1)
        if cfg.TRAIN.AUGMENT.CROP.KEEP_ONLY_CENTER_INSIDE:
            # consider boxes whose centers are inside
            x_ctr = (blobs['gt_boxes'][:, 0] + blobs['gt_boxes'][:, 2]) / 2
            y_ctr = (blobs['gt_boxes'][:, 1] + blobs['gt_boxes'][:, 3]) / 2
            inside_inds = np.where(
                np.logical_and.reduce(
                    (x_ctr >= width_start, x_ctr < width_start + width,
                     y_ctr >= height_start, y_ctr < height_start + height)))[0]
        else:
            # consider boxes who have positive overlaps
            inside_inds = np.where(
                np.logical_and(
                    np.clip(blobs['gt_boxes'][:, 0], width_start, None)<\
                        np.clip(blobs['gt_boxes'][:, 2], None, width_start+width),
                    np.clip(blobs['gt_boxes'][:, 1], height_start, None)<\
                        np.clip(blobs['gt_boxes'][:, 3], None, height_start+height)))[0]
        if (not cfg.TRAIN.AUGMENT.CROP.POSITIVE_ENFORCE
            ) or inside_inds.size > 0:
            flag = True
            break
    if not flag:
        # unfortunately, we didn't find any feasible cropping
        return blobs
    else:
        blobs['data'] = blobs['data'][..., height_start:height_start +
                                      height, width_start:width_start + width]
        blobs['gt_boxes'] = blobs['gt_boxes'][inside_inds]
        blobs['gt_boxes'][:, [0, 2]] -= width_start
        blobs['gt_boxes'][:, [0, 2]] = \
            np.clip(blobs['gt_boxes'][:, [0, 2]], 0.0, width)
        blobs['gt_boxes'][:, [1, 3]] -= height_start
        blobs['gt_boxes'][:, [1,3]] = \
            np.clip(blobs['gt_boxes'][:, [1,3]], 0.0, height)
        return blobs


def _project_im_rois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois = im_rois * im_scale_factor
    return rois


def _get_resize_range(roidb_x,
                      min_anchor=16.,
                      max_anchor=64.,
                      nbins=5,
                      nselected=3):
    assert nbins >= 2, 'there should be at least 2 bins'
    face_heights = roidb_x['boxes'][:, 3] - roidb_x['boxes'][:, 1]
    face_heights = face_heights[face_heights > 0]
    if face_heights.size == 0:
        return np.array([1.] * nbins)
    min_ratio = min_anchor / \
        np.max(face_heights)
    max_ratio = max_anchor / \
        np.min(face_heights)
    epsilon = cfg.EPS
    all_scales = np.exp(
        np.arange(
            np.log(min_ratio),
            np.log(max_ratio) + epsilon,
            (np.log(max_ratio) - np.log(min_ratio)) / (nbins - 1)))
    selected_scale = all_scales[(nbins - nselected) /
                                2:(nbins - nselected) / 2 + nselected]
    return selected_scale
