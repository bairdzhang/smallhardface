from __future__ import print_function
import cPickle
import os
import subprocess
import logging
logger = logging.getLogger(__name__)

import numpy as np
import scipy.sparse
from PIL import Image
import tarfile
import shutil

from utils.get_config import cfg
from datasets.imdb import imdb
from utils.tensorboard import tb


class general(imdb):
    def __init__(self, split):
        imdb.__init__(self, 'general_' + split)
        self._extension = split
        self._classes = ['bg', 'face']
        self._imgs_path = cfg.DATA_DIR
        self._image_paths = []
        for root, dirs, files in os.walk(cfg.DATA_DIR):
            for file in files:
                if file.endswith(".{}".format(split)):
                    self._image_paths.append(os.path.join(root, file))
        self._image_index = range(len(self._image_paths))

    def image_path_at(self, i):
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        image_path = os.path.join(self._imgs_path, self._image_paths[index])
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def gt_roidb(self):
        raise NotImplementedError("Not supported yet!")
        return None

    def write_detections(self, all_boxes, output_dir='./output/'):
        logger.info(
            'Writing the detections to text files: {}...'.format(output_dir))
        for i in range(len(self._image_paths)):
            img_path = self._image_paths[i]

            img_name = os.path.basename(img_path)
            img_dir = img_path[:img_path.find(img_name) - 1]
            if img_dir[0] == '/':
                img_dir = img_dir[1:]

            txt_fname = os.path.join(output_dir, img_dir,
                                     img_name.replace(self._extension, 'txt'))

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
        self.write_detections(all_boxes, output_dir)
        return "Detection results wrote to {}".format(output_dir)
