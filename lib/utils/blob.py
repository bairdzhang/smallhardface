# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
"""Blob helper functions."""

import cv2
import numpy as np
import numpy.random as npr

from utils.get_config import cfg


def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], im.shape[2]),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob


def prep_im_for_blob(im,
                     pixel_means,
                     target_size,
                     max_size,
                     mode="SHORT_SIDE",
                     face_median=0):
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    if cfg.TRAIN.AUGMENT.ENABLE:
        im = _distortion(im)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    if cfg.TRAIN.ORIG_SIZE or target_size < 0:
        im_scale = 1
    else:
        if mode == "SHORT_SIDE":
            im_scale = float(target_size) / float(im_size_min)
        elif mode == "FACE_AREA":
            im_scale = np.sqrt(float(target_size) / float(face_median))
        else:
            raise NotImplementedError(
                'Unknown mode in prep_im_for_blob: {}'.format(mode))

    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(
        im,
        None,
        None,
        fx=im_scale,
        fy=im_scale,
        interpolation=cv2.INTER_LINEAR)
    return im, im_scale


def _distortion(im):
    im = _brightness(im)
    if npr.randint(2):
        im = _contrast(im)
        hsv_im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        hsv_im = _saturation(hsv_im)
        hsv_im = _hue(hsv_im)
        im = cv2.cvtColor(hsv_im, cv2.COLOR_HSV2BGR)
    else:
        hsv_im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        hsv_im = _saturation(hsv_im)
        hsv_im = _hue(hsv_im)
        im = cv2.cvtColor(hsv_im, cv2.COLOR_HSV2BGR)
        im = _contrast(im)
    im = np.clip(np.round(im), 0.0, 255.0)
    return im


def _brightness(im):  # inplace transformation
    if npr.rand() < cfg.TRAIN.AUGMENT.BRIGHTNESS.PROB:
        delta = npr.uniform(-cfg.TRAIN.AUGMENT.BRIGHTNESS.DELTA,
                            cfg.TRAIN.AUGMENT.BRIGHTNESS.DELTA)
        im = np.clip(im + delta, 0.0, 255.0)
    return im


def _contrast(im):  # inplace transformation
    if npr.rand() < cfg.TRAIN.AUGMENT.CONTRAST.PROB:
        alpha = npr.uniform(cfg.TRAIN.AUGMENT.CONTRAST.LOWER,
                            cfg.TRAIN.AUGMENT.CONTRAST.UPPER)
        im = np.clip(im * alpha, 0.0, 255.0)
    return im


def _saturation(hsv_im):  # inplace transformation
    if npr.rand() < cfg.TRAIN.AUGMENT.SATURATION.PROB:
        alpha = npr.uniform(cfg.TRAIN.AUGMENT.SATURATION.LOWER,
                            cfg.TRAIN.AUGMENT.SATURATION.UPPER)
        hsv_im[..., 1] = np.clip(hsv_im[..., 1] * alpha, 0.0, 1.0)
    return hsv_im


def _hue(hsv_im):  # inplace transformation
    if npr.rand() < cfg.TRAIN.AUGMENT.HUE.PROB:
        delta = npr.uniform(-cfg.TRAIN.AUGMENT.HUE.DELTA,
                            cfg.TRAIN.AUGMENT.HUE.DELTA)
        hsv_im[..., 0] = (hsv_im[..., 0] + delta) % 360.0
    return hsv_im
