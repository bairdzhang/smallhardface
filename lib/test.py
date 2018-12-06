'''
Multi-GPU Test Code
'''
from __future__ import print_function
from multiprocessing import Process, Queue
import cPickle
import os
import sys
import cv2
import numpy as np
import logging
logger = logging.getLogger(__name__)

from utils.get_config import cfg
from utils.test_utils import _get_image_blob, _compute_scaling_factor
from utils.timer import Timer
from utils.tensorboard import tb
import caffe


def forward_net(net, blob, im_scale, pyramid=False, flip=False):
    """
    :param net: the trained network
    :param blob: a dictionary containing the image
    :param im_scale: the scale used for resizing the input image
    :param pyramid: whether using pyramid testing or not
    :return: the network outputs probs and pred_boxes (the probability of face/bg and the bounding boxes)
    """
    # Adding im_info to the data blob
    blob['im_info'] = np.array(
        [[blob['data'].shape[2], blob['data'].shape[3], im_scale]],
        dtype=np.float32)

    h, w = blob['data'].shape[2:]
    new_h = int(np.ceil(1.0 * h / cfg.MAX_RESOLUTION) * cfg.MAX_RESOLUTION)
    new_w = int(np.ceil(1.0 * w / cfg.MAX_RESOLUTION) * cfg.MAX_RESOLUTION)
    data = np.pad(blob['data'], ((0, 0), (0, 0), (0, new_h - h),
                                 (0, new_w - w)), 'constant')

    # Reshape network inputs
    net.blobs['data'].reshape(*(data.shape))
    net.blobs['im_info'].reshape(*(blob['im_info'].shape))

    # Forward the network
    net_args = {
        'data': data.astype(np.float32, copy=False),
        'im_info': blob['im_info'].astype(np.float32, copy=False)
    }

    blobs_out = net.forward(**net_args)

    if flip:
        for i in filter(lambda x: x.startswith('boxes'), blobs_out.keys()):
            blobs_out[i][:, [1, 3]] = w - blobs_out[i][:, [3, 1]]

    if pyramid:
        pred_boxes = []
        probs = []
        if 'boxes' in net.blobs:
            cur_boxes = net.blobs['boxes'].data
            # unscale back to raw image space
            cur_boxes = cur_boxes[:, 1:5] / im_scale
            # Repeat boxes
            cur_probs = net.blobs['cls_prob'].data
            pred_boxes.append(np.tile(cur_boxes, (1, cur_probs.shape[1])))
            probs.append(cur_probs)
        else:
            level_suffixs = map(
                lambda x: x.split('_')[-1],
                filter(lambda x: x.startswith('boxes'), net.blobs.keys()))
            if len(cfg.TEST.LEVEL) > 0:
                logger.warning(
                    'Subset of levels selected for evaluation: {}'.format(
                        cfg.TEST.LEVEL))
                level_suffixs = cfg.TEST.LEVEL
            for level in level_suffixs:
                cur_boxes = net.blobs['boxes_{}'.format(level)].data
                # unscale back to raw image space
                cur_boxes = cur_boxes[:, 1:5] / im_scale
                # Repeat boxes
                cur_probs = net.blobs['cls_prob_{}'.format(level)].data
                pred_boxes.append(np.tile(cur_boxes, (1, cur_probs.shape[1])))
                probs.append(cur_probs)
    else:
        pred_boxes = []
        probs = []
        if 'boxes' in net.blobs:
            raise NotImplementedError("Please complete this part!")
        else:
            level_suffixs = map(
                lambda x: x.split('_')[-1],
                filter(lambda x: x.startswith('boxes'), net.blobs.keys()))
            if len(cfg.TEST.LEVEL) > 0:
                logger.warning(
                    'Subset of levels selected for evaluation: {}'.format(
                        cfg.TEST.LEVEL))
                level_suffixs = cfg.TEST.LEVEL
            for level in level_suffixs:
                cur_boxes = net.blobs['boxes_{}'.format(level)].data
                # unscale back to raw image space
                cur_boxes = cur_boxes[:, 1:5] / im_scale
                # Repeat boxes
                cur_probs = net.blobs['cls_prob_{}'.format(level)].data
                pred_boxes.append(np.tile(cur_boxes, (1, cur_probs.shape[1])))
                probs.append(cur_probs)
    return probs, pred_boxes


def detect(net, im_path, thresh=0.05, timers=None, pyramid=False):
    if not timers:
        timers = {'detect': Timer(), 'misc': Timer()}

    im = cv2.imread(im_path)
    imfname = os.path.basename(im_path)
    sys.stdout.flush()
    timers['detect'].tic()

    if not pyramid:
        im_scale = _compute_scaling_factor(im.shape, cfg.TEST.SCALES[0],
                                           cfg.TEST.MAX_SIZE)
        im_blob = _get_image_blob(im, [im_scale])
        probs, boxes = forward_net(net, im_blob[0], im_scale, pyramid=False)
        if isinstance(probs, list):
            probs = np.vstack(probs)
            boxes = np.vstack(boxes)
        boxes = boxes[:, 0:4]
    else:
        all_probs = []
        all_boxes = []
        # Compute the scaling coefficients for the pyramid
        base_scale = _compute_scaling_factor(im.shape,
                                             cfg.TEST.PYRAMID_BASE_SIZE[0],
                                             cfg.TEST.PYRAMID_BASE_SIZE[1])
        pyramid_scales = [
            float(scale) / cfg.TEST.PYRAMID_BASE_SIZE[0] * base_scale
            for scale in cfg.TEST.SCALES
        ]

        im_blobs = _get_image_blob(im, pyramid_scales)

        for i in range(len(pyramid_scales)):
            probs, boxes = forward_net(
                net, im_blobs[i], pyramid_scales[i], pyramid=True)
            for j in xrange(len(probs)):
                all_boxes.append(boxes[j][:, 0:4])
                all_probs.append(probs[j].copy())
            if cfg.TEST.FLIP:
                probs, boxes = forward_net(
                    net, {'data': im_blobs[i]['data'][..., ::-1]},
                    pyramid_scales[i],
                    pyramid=True,
                    flip=True)
                for j in xrange(len(probs)):
                    all_boxes.append(boxes[j][:, 0:4])
                    all_probs.append(probs[j].copy())

        probs = np.concatenate(all_probs)
        boxes = np.concatenate(all_boxes)
    timers['detect'].toc()
    timers['misc'].tic()
    cls_dets = [None] * (probs.shape[1] - 1)
    for class_i in range(1, probs.shape[1]):
        inds = np.where(probs[:, class_i] > thresh)[0]
        probs_i = probs[inds, class_i]
        boxes_i = boxes[inds, :]
        dets = np.hstack((boxes_i, probs_i[:, np.newaxis])) \
            .astype(np.float32, copy=False)
        if cfg.TEST.NMS_METHOD == "BBOX_VOTE":
            cls_dets[class_i - 1] = bbox_vote(dets)
        elif cfg.TEST.NMS_METHOD == "NMS":
            keep = nms(dets, cfg.TEST.NMS_THRESH)
            cls_dets[class_i - 1] = dets[keep, :]
        else:
            raise NotImplementedError("Unknown NMS method: {}".format(
                cfg.TEST.NMS_METHOD))
    assert all([_ is not None for _ in cls_dets]), 'None in detection results'
    timers['misc'].toc()
    return cls_dets, timers


def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    if det.shape[0] == 0:
        dets = np.array([[10, 10, 20, 20, 0.0001]])
        det = np.empty(shape=[0, 5])
    while det.shape[0] > 0:
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)
        merge_index = np.where(o >= cfg.TEST.NMS_THRESH)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)
        if merge_index.shape[0] <= 1:
            if det.shape[0] == 0:
                try:
                    dets = np.row_stack((dets, det_accu))
                except:
                    dets = det_accu
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(
            det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum
    return dets


def inference_worker(rank,
                     imdb,
                     target_test,
                     start,
                     end,
                     thresh,
                     result_queue=None):
    # Loading the network
    cfg.GPU_ID = cfg.TEST.GPU_ID[rank]
    caffe.set_mode_gpu()
    caffe.set_device(cfg.GPU_ID)
    net = caffe.Net(str(target_test), str(cfg.TEST.MODEL), caffe.TEST)

    timers = {'detect': Timer(), 'misc': Timer()}

    pyramid = True if len(cfg.TEST.SCALES) > 1 else False

    dets = [[[] for _ in range(start, end)] for _ in range(imdb.num_classes)]

    for i in range(start, end):
        im_path = imdb.image_path_at(i)
        dets_, detect_time = detect(
            net, im_path, thresh, timers=timers, pyramid=pyramid)
        for _ in range(imdb.num_classes - 1):
            dets[_ + 1][i - start] = dets_[_]
        if rank == 0:
            try:
                tb.sess.add_scalar_value(
                    "detect-time",
                    timers['detect'].average_time,
                    step=i - start)
                tb.sess.add_scalar_value(
                    "misc-time", timers['misc'].average_time, step=i - start)
                print(
                    '\r{:02d}% detect-time: {:.3f}s, misc-time:{:.3f}s, remain-time: {:.3f}s'
                    .format(
                        int(100 * (i + 1 - start) / (end - start)),
                        timers['detect'].average_time,
                        timers['misc'].average_time,
                        (end - i - 1) * (timers['detect'].average_time +
                                         timers['misc'].average_time)),
                    end='')
            except:
                logger.warning('Failed to submit data to Tensorboard')
    if result_queue:
        result_queue.put((rank, dets))
        return
    return dets


def demo(target_test, thresh):
    # Loading the network
    cfg.GPU_ID = cfg.TEST.GPU_ID[0]
    caffe.set_mode_gpu()
    caffe.set_device(cfg.GPU_ID)
    net = caffe.Net(str(target_test), str(cfg.TEST.MODEL), caffe.TEST)
    pyramid = True if len(cfg.TEST.SCALES) > 1 else False
    im_path = cfg.TEST.DEMO.IMAGE
    dets, detect_time = detect(
        net, im_path, thresh, timers=None, pyramid=pyramid)
    im = cv2.imread(cfg.TEST.DEMO.IMAGE)
    for i in range(dets[0].shape[0]):
        if dets[0][i, -1] < thresh:
            continue
        cv2.rectangle(im, (int(dets[0][i, 0]), int(dets[0][i, 1])),
                      (int(dets[0][i, 2]), int(dets[0][i, 3])), (0, 255, 0), 2)
    cv2.imwrite('/tmp/demo_res.jpg', im)
    return None


def test_net(imdb,
             output_dir,
             target_test,
             thresh=0.05,
             no_cache=False,
             step=0):
    # Run demo
    if imdb is None:
        assert cfg.TEST.DEMO.ENABLE, "check your config and stderr!"
        return demo(target_test, thresh)
    # Initializing the timers
    logger.info('Evaluating {} on {}'.format(cfg.NAME, imdb.name))

    run_inference = True
    if not no_cache:
        det_file = os.path.join(output_dir, 'detections.pkl')
        if os.path.exists(det_file):
            try:
                with open(det_file, 'r') as f:
                    dets = cPickle.load(f)
                    run_inference = False
                    logger.info(
                        'Loading detections from cache: {}'.format(det_file))
            except:
                logger.warning(
                    'Could not load the cached detections file, detecting from scratch!'
                )

    # Perform inference on images if necessary
    if run_inference:
        pyramid = True if len(cfg.TEST.SCALES) > 1 else False
        if isinstance(cfg.TEST.GPU_ID, int):
            cfg.TEST.GPU_ID = [cfg.TEST.GPU_ID]
        assert len(cfg.TEST.GPU_ID) >= 1, "You must specify at least one GPU"
        if len(cfg.TEST.GPU_ID) == 1:
            dets = inference_worker(0, imdb, target_test, 0, len(imdb), thresh)
        else:
            result_queue = Queue()
            procs = []
            len_per_gpu = int(np.ceil(1. * len(imdb) / len(cfg.TEST.GPU_ID)))
            for rank in range(len(cfg.TEST.GPU_ID)):
                p = Process(
                    target=inference_worker,
                    args=(rank, imdb, target_test, len_per_gpu * rank,
                          min(len_per_gpu * (rank + 1), len(imdb)), thresh,
                          result_queue))
                p.daemon = True
                p.start()
                procs.append(p)
            dets = [result_queue.get() for _ in procs]
            for p in procs:
                p.join()
            dets = [det[1] for det in sorted(dets, key=lambda x: x[0])]
            dets = [[_ for det in dets for _ in det[i]]
                    for i in range(imdb.num_classes)]
        assert len(dets[0]) == len(imdb), "Detection result compromised"
        det_file = os.path.join(output_dir, 'detections.pkl')
        if not no_cache:
            with open(det_file, 'wb') as f:
                cPickle.dump(dets, f, cPickle.HIGHEST_PROTOCOL)

    # Evaluate the detections
    logger.info('Evaluating detections')
    result = imdb.evaluate_detections(
        all_boxes=dets, output_dir=output_dir, method_name=cfg.NAME, step=step)
    logger.info(result)
    logger.info('All Done!')


def get_testing_roidb(imdb):
    """
    Get the testing roidb given an imdb
    :param imdb: The testing imdb
    :return: The testing roidb
    """

    logger.info('Preparing testing data...')
    # Add required information to imdb
    imdb.prepare_roidb()
    # Filter the roidb
    logger.info('done')
    return imdb.roidb
