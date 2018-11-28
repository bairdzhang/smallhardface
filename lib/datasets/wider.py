from __future__ import print_function
import cPickle
import os
import subprocess
import logging
logger = logging.getLogger(__name__)

import numpy as np
import scipy.sparse
from PIL import Image
import os
import tarfile
import shutil

from utils.get_config import cfg
from datasets.imdb import imdb
from wider_eval_tools.wider_eval import wider_eval
from utils.tensorboard import tb


class wider(imdb):
    def __init__(self, split):
        self._test_flag = True if split == 'test' else False
        self._split = split
        imdb.__init__(self, 'wider_' + split)
        self._image_set = split


        self._annotation_file_name = 'wider_face_test_filelist.txt' if self._test_flag else \
            'wider_face_{}_bbx_gt.txt'.format(split)

        self._dataset_path = cfg.DATA_DIR
        self._imgs_path = os.path.join(self._dataset_path,
                                       'WIDER_{}'.format(split), 'images')

        # Read the annotations file
        anno_path = os.path.join(self._dataset_path, 'wider_face_split',
                                 self._annotation_file_name)
        assert os.path.isfile(
            anno_path), 'Annotation file not found {}'.format(anno_path)
        self._fp_bbox_map = {}
        with open(anno_path, 'r') as file:
            annos = file.readlines()

        self._fp_bbox_map = {}
        count = 0
        if not self._test_flag:
            while count < len(annos):
                name = str(annos[count]).rstrip()
                self._fp_bbox_map[name] = []
                count += 1
                n_anno = int(annos[count])
                for i in xrange(n_anno):
                    count += 1
                    bbox = annos[count].split(' ')[0:4]
                    bbox = [int(round(float(x))) for x in bbox]
                    x1 = max(0, bbox[0])
                    y1 = max(0, bbox[1])
                    self._fp_bbox_map[name].append(
                        [x1, y1, x1 + bbox[2], y1 + bbox[3]])
                count += 1
            self._image_paths = self._fp_bbox_map.keys()
        else:
            self._image_paths = []
            for path in annos:
                self._image_paths.append(str(path).rstrip())

        self._image_index = range(len(self._image_paths))
        self._classes = ['bg', 'face']

    def image_path_at(self, i):
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        image_path = os.path.join(self._imgs_path, self._image_paths[index])
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def gt_roidb(self):
        cache_file = os.path.join(
            self.cache_path, '{}_{}_gt_roidb.pkl'.format(
                self.name, self._split))
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            logger.info('{} gt roidb loaded from {}'.format(
                self.name, cache_file))
            return roidb

        roidb = []

        for fp in self._image_paths:
            if self._test_flag:
                roidb.append({
                    'image_size':
                    Image.open(os.path.join(self._imgs_path, fp)).size,
                    'file_path':
                    os.path.join(self._imgs_path, fp)
                })
            else:
                boxes = np.zeros([len(self._fp_bbox_map[fp]), 4], np.float)

                gt_classes = np.ones([len(self._fp_bbox_map[fp])], np.int32)
                overlaps = np.zeros([len(self._fp_bbox_map[fp]), 2], np.float)

                ix = 0

                for bbox in self._fp_bbox_map[fp]:
                    imsize = Image.open(os.path.join(self._imgs_path, fp)).size

                    x1 = bbox[0]
                    y1 = bbox[1]
                    x2 = min(imsize[0], bbox[2])
                    y2 = min(imsize[1], bbox[3])

                    if (x2 - x1) < 1 or y2 - y1 < 1:
                        continue

                    boxes[ix, :] = np.array([x1, y1, x2, y2], np.float)

                    cls = int(1)
                    gt_classes[ix] = cls
                    overlaps[ix, cls] = 1.0
                    ix += 1
                overlaps = scipy.sparse.csr_matrix(overlaps)

                roidb.append({
                    'boxes': boxes,
                    'gt_classes': gt_classes,
                    'gt_overlaps': overlaps,
                    'flipped': False,
                    'image_size': imsize,
                    'file_path': os.path.join(self._imgs_path, fp)
                })

        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        logger.info('wrote gt roidb to {}'.format(cache_file))

        return roidb

    def write_detections(self, all_boxes, output_dir='./output/'):

        logger.info(
            'Writing the detections to text files: {}...'.format(output_dir))
        for i in range(len(self._image_paths)):
            img_path = self._image_paths[i]

            img_name = os.path.basename(img_path)
            img_dir = img_path[:img_path.find(img_name) - 1]

            txt_fname = os.path.join(output_dir, img_dir,
                                     img_name.replace('jpg', 'txt'))

            res_dir = os.path.join(output_dir, img_dir)
            if not os.path.isdir(res_dir):
                os.makedirs(res_dir)

            with open(txt_fname, 'w') as f:
                f.write(img_path + '\n')
                f.write(str(len(all_boxes[1][i])) + '\n')
                for det in all_boxes[1][i]:
                    f.write(
                        '%d %d %d %d %g \n' %
                        (int(det[0]), int(det[1]), int(det[2]) - int(det[0]),
                         int(det[3]) - int(det[1]), det[4]))
        logger.info('Done!')

    def evaluate_detections(self,
                            all_boxes,
                            output_dir='./output/',
                            method_name='smallhard',
                            step=0):
        detections_txt_path = os.path.join(output_dir, 'detections')
        self.write_detections(all_boxes, detections_txt_path)

        logger.info('Evaluating detections using official WIDER toolbox...')
        ap, pr = wider_eval(
            detections_txt_path,
            os.path.join(cfg.DATA_DIR, 'ground_truth'),
            mimic_eval_bug=cfg.MISC.MIMIC_EVAL_BUG,
            IoU_thresh=cfg.TEST.IOU_THRESH)
        with tarfile.open(os.path.join(output_dir, 'result.tar.gz'),
                          'w:gz') as tar:
            tar.add(
                detections_txt_path,
                arcname=os.path.basename(detections_txt_path))
        shutil.rmtree(detections_txt_path)

        tb.sess.add_scalar_value("easy", ap[0], step=step)
        tb.sess.add_scalar_value("medium", ap[1], step=step)
        tb.sess.add_scalar_value("hard", ap[2], step=step)
        result = 'Easy: {:.4f}, Medium: {:.4f}, Hard: {:.4f}'.format(*ap)
        return result
