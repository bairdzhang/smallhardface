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


class afw(imdb):
    def __init__(self, split):
        imdb.__init__(self, 'afw_' + split)
        self._classes = ['bg', 'face']
        self._dataset_path = cfg.DATA_DIR
        self._imgs_path = cfg.DATA_DIR
        list_file = os.path.join(self._dataset_path, 'afw_img_list.txt')
        with open(list_file, 'r') as file:
            file_list = file.readlines()
        self._image_paths = [x.strip() for x in file_list]
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
        raise NotImplementedError("Not supported yet!")
        return None

    def write_detections(self, all_boxes, output_dir='./output/'):
        logger.info(
            'Writing the detections to text files: {}...'.format(output_dir))
        txt_fname = os.path.join(output_dir, 'afw_res.txt')
        with open(txt_fname, 'w') as f:
            for i in range(len(self._image_paths)):
                img_path = self._image_paths[i]
                img_name = os.path.basename(img_path)
                img_name = os.path.splitext(img_name)[0]
                for res in all_boxes[1][i]:
                    score = res[-1]
                    xmin, ymin, xmax, ymax = res[:4]
                    ymin += 0.2 * (ymax - ymin + 1)
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(
                        img_name, res[-1], xmin, ymin, xmax, ymax))
        logger.info('Done!')

    def evaluate_detections(self,
                            all_boxes,
                            output_dir='./output/',
                            method_name='smallhard',
                            step=0):
        self.write_detections(all_boxes, output_dir)
        return "Detection results wrote to {}".format(output_dir)
