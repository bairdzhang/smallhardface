# -----------------------------------------------------------
# Entrance for training and testing
# -----------------------------------------------------------
import os, sys
if not 'caffe/python' in sys.path:
    sys.path.insert(0, 'caffe/python')
if not 'lib' in sys.path:
    sys.path.insert(0, 'lib')
from utils.get_config import cfg, cfg_from_file, cfg_from_list, get_output_dir, cfg_print, cfg_dump, cfg_table
from train import train_net, get_training_roidb
from test import test_net
import argparse
import sys
import os.path as osp
import numpy as np
import re
import datetime
import glob
from datasets.factory import get_imdb
from prototxt.manipulate import manipulate_solver, manipulate_train, manipulate_test
from utils.tensorboard import tb, Tensorboard, TBExp
import logging
logging.basicConfig(
    format=
    '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%m-%d-%Y:%H:%M:%S',
    level=logging.DEBUG if
    ('DEBUG' in os.environ and os.environ['DEBUG'] == '1') else logging.INFO)
logger = logging.getLogger(__name__)


def parser():
    parser = argparse.ArgumentParser(
        'Train and test', description='Give settings')
    parser.add_argument(
        '--train', dest='train', help='do training', default='true')
    parser.add_argument(
        '--test', dest='test', help='do testing', default='true')
    parser.add_argument(
        '--conf', dest='conf_file', help='provide configure file', default='')
    parser.add_argument(
        '--amend',
        dest='set_cfgs',
        help='provide amend cfgs',
        default=None,
        nargs=argparse.REMAINDER)
    return parser.parse_args()


if __name__ == '__main__':
    args = parser()

    # Load settings
    if args.conf_file:
        cfg_from_file(args.conf_file)

    # For train and test, usually we do not need cache; unless overridden by amend
    cfg.TEST.NO_CACHE = True
    if args.set_cfgs:
        cfg_from_list(args.set_cfgs)

    # Record logs into cfg
    cfg.LOG.CMD = ' '.join(sys.argv)
    cfg.LOG.TIME = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    np.random.seed(int(cfg.RNG_SEED))

    if cfg.TENSORBOARD.ENABLE:
        tb.client = Tensorboard(
            hostname=cfg.TENSORBOARD.HOSTNAME, port=cfg.TENSORBOARD.PORT)
        tb.sess = tb.client.create_experiment(cfg.NAME + '_' + cfg.LOG.TIME)

    if args.train == 'true' or args.train == 'True':  # the training entrance
        # Get training imdb
        imdb = get_imdb(cfg.TRAIN.DB)
        roidb = get_training_roidb(imdb)

        # Redirect stderr
        output_dir = get_output_dir(imdb.name, cfg.NAME + '_' + cfg.LOG.TIME)
        f = open(osp.join(output_dir, 'stderr.log'), 'w', 0)
        os.dup2(f.fileno(), sys.stderr.fileno())
        os.dup2(sys.stderr.fileno(), sys.stderr.fileno())

        # Edit solver and train prototxts
        target_sw = osp.join(output_dir, 'solver.prototxt')
        target_train = osp.join(output_dir, 'train.prototxt')

        manipulate_solver(cfg.TRAIN.SOLVER, target_sw, train_net=target_train)
        manipulate_train(cfg.TRAIN.PROTOTXT, target_train)

        if isinstance(cfg.TRAIN.GPU_ID, int):
            cfg.TRAIN.GPU_ID = [cfg.TRAIN.GPU_ID]

        cfg_print(cfg)

        with open(osp.join(output_dir, 'cfgs.txt'), 'w') as f:
            cfg_dump({i: cfg[i] for i in cfg if i != 'TEST'}, f)
        tb.sess.add_text('train_cfg', \
                         cfg_table({i: cfg[i] for i in cfg if i != 'TEST'}))
        train_net(
            target_sw,
            roidb,
            output_dir=output_dir,
            pretrained_model=cfg.TRAIN.PRETRAINED,
            max_iter=cfg.TRAIN.ITERS,
            gpus=cfg.TRAIN.GPU_ID)

        f.close()
        # Set test models for the following testing
        cfg.TEST.MODEL = osp.join(output_dir, 'final.caffemodel')

    if args.test == 'true' or args.test == 'True':  # the testing entrance
        if isinstance(cfg.TEST.GPU_ID, int):
            cfg.TEST.GPU_ID = [cfg.TEST.GPU_ID]

        if not cfg.TEST.DEMO.ENABLE:
            imdb = get_imdb(cfg.TEST.DB)
            output_dir = get_output_dir(imdb.name, cfg.NAME + '_' + cfg.LOG.TIME)
        else:
            imdb = None
            output_dir = get_output_dir("demo", cfg.NAME + '_' + cfg.LOG.TIME)

        f = open(osp.join(output_dir, 'stderr.log'), 'w', 0)
        os.dup2(f.fileno(), sys.stderr.fileno())
        os.dup2(sys.stderr.fileno(), sys.stderr.fileno())

        # Edit test prototxts
        target_test = osp.join(output_dir, 'test.prototxt')

        manipulate_test(cfg.TEST.PROTOTXT, target_test)

        with open(osp.join(output_dir, 'cfgs.txt'), 'w') as f:
            cfg_dump({i: cfg[i] for i in cfg if i != 'TRAIN'}, f)
        tb.sess.add_text('test_cfg', \
                         cfg_table({i: cfg[i] for i in cfg if i != 'TRAIN'}))

        test_net(imdb, output_dir, target_test, no_cache=cfg.TEST.NO_CACHE)
        f.close()
