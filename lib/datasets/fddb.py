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
from utils.tensorboard import tb

_FDDB_RECT = 0
_FDDB_ELLI = 1

_FDDB_EVAL_COMMAND = """
/{root}/evaluation/evaluate -a /{root}/FDDB-folds/val_gt.txt -i /{root}/ -l /{root}/FDDB-folds/val.txt -d {detect} -f {format} -r {output_dir}
"""


class fddb(imdb):
    def __init__(self, split):
        self._test_flag = True if split == 'test' else False
        self._split = split
        imdb.__init__(self, 'fddb_' + split)
        self._image_set = split
        self._dataset_path = cfg.DATA_DIR
        self._imgs_path = cfg.DATA_DIR

        list_file = os.path.join(self._dataset_path,
                                 'FDDB-folds/{}.txt'.format(split))
        with open(list_file, 'r') as file:
            file_list = file.readlines()
        self._image_paths = [x.strip() + '.jpg' for x in file_list]
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

    def write_detections_rect(self, all_boxes, output_dir='./output/'):
        logger.info(
            'Writing the detections to text files: {}...'.format(output_dir))
        with open(os.path.join(output_dir, 'detection_rect.txt'), 'w') as f:
            for i in range(len(self._image_paths)):
                img_path = self._image_paths[i]
                img_name = os.path.splitext(img_path)[0]
                f.write('{:s}\n'.format(img_name))
                num_detections = all_boxes[1][i].shape[0]
                f.write('{:d}\n'.format(num_detections))
                for j in range(num_detections):
                    f.write('{:.3f} {:.3f} {:.3f} {:.3f} {:.10f}\n'.format(
                        all_boxes[1][i][j][0], all_boxes[1][i][j][1],
                        all_boxes[1][i][j][2] - all_boxes[1][i][j][0] + 1,
                        all_boxes[1][i][j][3] - all_boxes[1][i][j][1] + 1,
                        all_boxes[1][i][j][4]))
        logger.info('Done!')

    def evaluate_detections(self,
                            all_boxes,
                            output_dir='./output/',
                            method_name='smallhard',
                            step=0):
        self.write_detections_rect(all_boxes, output_dir)
        process_rect = subprocess.Popen(
            _FDDB_EVAL_COMMAND.format(
                root=cfg.DATA_DIR,
                detect=os.path.join(output_dir, 'detection_rect.txt'),
                format=_FDDB_RECT,
                output_dir=output_dir + '/rect_'),
            shell=True,
            stdout=subprocess.PIPE)
        process_rect.wait()
        with open(os.path.join(output_dir, 'rect_DiscROC.txt'), 'r') as f:
            lines = f.readlines()
        disc_res = np.array(map(lambda x: x.strip().split(),
                                lines)).astype(np.float)
        rect_disc_at_1000 = disc_res[np.where(disc_res[:, 1] < 1000)[0][0], 0]
        with open(os.path.join(output_dir, 'rect_ContROC.txt'), 'r') as f:
            lines = f.readlines()
        cont_res = np.array(map(lambda x: x.strip().split(),
                                lines)).astype(np.float)
        rect_cont_at_1000 = cont_res[np.where(cont_res[:, 1] < 1000)[0][0], 0]
        tb.sess.add_scalar_value(
            "rect_disc_at_1000", rect_disc_at_1000, step=step)
        tb.sess.add_scalar_value(
            "rect_cont_at_1000", rect_cont_at_1000, step=step)
        return "rect_disc_at_1000: {:.4f}, rect_cont_at_1000: {:.4f}".format(
            rect_disc_at_1000, rect_cont_at_1000)
