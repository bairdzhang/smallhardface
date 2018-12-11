from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf
from utils.get_config import cfg
from caffe.draw import draw_net_to_file
from utils.tensorboard import tb
import yaml
import logging
import numpy as np
import copy
logger = logging.getLogger(__name__)


def manipulate_solver(ori, target_sw, train_net=None):
    sw_pb = caffe_pb2.SolverParameter()
    with open(ori, 'r') as f:
        solver_txt = f.read()
        txtf.Merge(solver_txt, sw_pb)
    sw_pb.iter_size = cfg.TRAIN.ITERSIZE
    sw_pb.base_lr = cfg.TRAIN.LR.BASELR
    sw_pb.weight_decay = cfg.TRAIN.WEIGHT_DECAY
    if train_net:
        sw_pb.train_net = train_net
    if cfg.TRAIN.LR_POLICY == "STEP":
        sw_pb.lr_policy = u'step'
        sw_pb.stepsize = cfg.TRAIN.STEPSIZE
    elif cfg.TRAIN.LR_POLICY == "MULTISTEP":
        sw_pb.lr_policy = u'multistep'
        sw_pb.ClearField('stepvalue')
        for sv in cfg.TRAIN.STEPVALUE:
            sw_pb.stepvalue.append(sv)
    with open(target_sw, 'w') as f:
        f.write(str(sw_pb))


def manipulate_train(ori, target_train, **kwargs):
    train_pb = caffe_pb2.NetParameter()
    if cfg.MODEL.DIFFERENT_DILATION.ENABLE:
        ori = 'models/train_different_dilation_template.prototxt'
        with open(ori, 'r') as f:
            train_txt = f.read()
            txtf.Merge(train_txt, train_pb)
        train_pb = _add_dimension_reduction(train_pb)
        train_pb = _apply_mult_lr(train_pb)
        train_vis_file = '.'.join(target_train.rsplit('.')[:-1]) + '.jpg'
        with open(target_train, 'w') as f:
            f.write(str(train_pb))
        draw_net_to_file(train_pb, train_vis_file, 'LR', 0)
        tb.sess.add_image('train_net', train_vis_file, wall_time=0, step=0)
        return None

    with open(ori, 'r') as f:
        train_txt = f.read()
        txtf.Merge(train_txt, train_pb)
    train_pb = _add_dimension_reduction(train_pb)
    train_pb = _apply_mult_lr(train_pb)
    train_vis_file = '.'.join(target_train.rsplit('.')[:-1]) + '.jpg'
    with open(target_train, 'w') as f:
        f.write(str(train_pb))
    draw_net_to_file(train_pb, train_vis_file, 'LR', 0)
    tb.sess.add_image('train_net', train_vis_file, wall_time=0, step=0)


def manipulate_test(ori, target_test, **kwargs):  # TODO more elegant editing
    test_pb = caffe_pb2.NetParameter()
    if cfg.MODEL.DIFFERENT_DILATION.ENABLE:
        ori = 'models/test_different_dilation_template.prototxt'
        with open(ori, 'r') as f:
            test_txt = f.read()
            txtf.Merge(test_txt, test_pb)
        test_pb = _add_dimension_reduction(test_pb)
        test_vis_file = '.'.join(target_test.rsplit('.')[:-1]) + '.jpg'
        with open(target_test, 'w') as f:
            f.write(str(test_pb))
        draw_net_to_file(test_pb, test_vis_file, 'LR', 1)
        tb.sess.add_image('test_net', test_vis_file, wall_time=0, step=0)
        return None

    with open(ori, 'r') as f:
        test_txt = f.read()
        txtf.Merge(test_txt, test_pb)
    test_pb = _add_dimension_reduction(test_pb)
    test_vis_file = '.'.join(target_test.rsplit('.')[:-1]) + '.jpg'
    with open(target_test, 'w') as f:
        f.write(str(test_pb))
    draw_net_to_file(test_pb, test_vis_file, 'LR', 1)
    tb.sess.add_image('test_net', test_vis_file, wall_time=0, step=0)


def _simple_conv_layer(name,
                       bottom,
                       top,
                       num_output,
                       kernel_size,
                       pad,
                       dilation=1,
                       std=0.01,
                       bias=0.0,
                       param_type=0):
    """
    param_type:
    0 -> do nothing
    1 -> (lr 1, decay 0; lr 2, decay 0)
    2 -> (lr 1, decay 1; lr 2, decay 0)
    3 -> (lr 10, decay 1; lr 20, decay 0)
    4 -> (lr 1, decay 1; lr 1, decay 1)
    """
    conv_layer = caffe_pb2.LayerParameter()
    conv_layer.name = name
    conv_layer.type = 'Convolution'
    conv_layer.bottom.append(bottom)
    conv_layer.top.append(top)
    conv_layer.convolution_param.num_output = num_output
    conv_layer.convolution_param.pad.append(pad)
    conv_layer.convolution_param.kernel_size.append(kernel_size)
    conv_layer.convolution_param.weight_filler.type = "gaussian"
    conv_layer.convolution_param.weight_filler.std = std
    conv_layer.convolution_param.bias_filler.type = "constant"
    conv_layer.convolution_param.bias_filler.value = bias
    conv_layer.convolution_param.dilation.append(dilation)
    conv_layer.ClearField('param')
    conv_layer.param.extend([caffe_pb2.ParamSpec()] * 2)
    if param_type == 1:
        conv_layer.param[0].lr_mult = 1.0
        conv_layer.param[0].decay_mult = 0.0
        conv_layer.param[1].lr_mult = 2.0
        conv_layer.param[1].decay_mult = 0.0
    elif param_type == 2:
        conv_layer.param[0].lr_mult = 1.0
        conv_layer.param[0].decay_mult = 1.0
        conv_layer.param[1].lr_mult = 2.0
        conv_layer.param[1].decay_mult = 0.0
    elif param_type == 3:
        conv_layer.param[0].lr_mult = 10.0
        conv_layer.param[0].decay_mult = 1.0
        conv_layer.param[1].lr_mult = 20.0
        conv_layer.param[1].decay_mult = 0.0
    elif param_type == 4:
        conv_layer.param[0].lr_mult = 1.0
        conv_layer.param[0].decay_mult = 1.0
        conv_layer.param[1].lr_mult = 2.0
        conv_layer.param[1].decay_mult = 1.0
    return conv_layer


def _simple_relu_layer(name, bottom, top=None):
    relu_layer = caffe_pb2.LayerParameter()
    relu_layer.name = name
    relu_layer.type = 'ReLU'
    relu_layer.bottom.append(bottom)
    relu_layer.top.append(top if top is not None else bottom)
    return relu_layer


def _apply_mult_lr(pb):
    split = np.min(
        np.where([x.name.startswith('head') for x in pb.layer])[0])
    for i, x in enumerate(pb.layer):
        for j in x.param:
            if i < split:
                j.lr_mult = j.lr_mult * cfg.TRAIN.LR.BACKBONE_MULT
            else:
                j.lr_mult = j.lr_mult * cfg.TRAIN.LR.HEAD_MULT
    return pb


def _add_dimension_reduction(pb):
    if not cfg.MODEL.DIFFERENT_DILATION.ENABLE:
        return pb
    split = np.min(
        np.where([x.name.startswith('head') for x in pb.layer])[0])
    assert pb.layer[split - 2].name == 'conv4_fuse_final'
    pb.layer[split - 2].top[0] += '_tmp'
    pb.layer[split - 1].bottom[0] += '_tmp'
    pb.layer[split - 1].top[0] += '_tmp'
    new_layers = pb.layer[:split] + [
        _simple_conv_layer(
            'conv4_fuse_final_dim_red',
            'conv4_fuse_final_tmp',
            'conv4_fuse_final',
            128,
            3,
            1,
            param_type=4),
        _simple_relu_layer('conv4_fuse_final_dim_red_relu', 'conv4_fuse_final')
    ] + pb.layer[split:]
    pb.ClearField('layer')
    pb.layer.extend(new_layers)
    return pb
