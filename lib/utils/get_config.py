from __future__ import print_function
import os
import os.path as osp
import toml
import numpy as np
from collections import OrderedDict
from easydict import EasyDict


def _sort_dict(d):
    res = OrderedDict(sorted(d.items()))

    def __sort(p):
        for i in p:
            if isinstance(p[i], dict):
                p[i] = OrderedDict(sorted(p[i].items()))
                __sort(p[i])

    __sort(res)
    return res


# PARSE THE DEFAULT CONFIG
_default_cfg_path = 'configs/default.toml'
assert osp.isfile(
    _default_cfg_path), 'The default config is not found in {}!'.format(
        _default_cfg_path)
_default_cfg = toml.load(_default_cfg_path)
_default_cfg.update({'LOG': {}})
cfg = EasyDict(_sort_dict(_default_cfg))

# Keep PIXEL_MEANS as list for compatibility of TOML
# cfg.PIXEL_MEANS = np.array(cfg.PIXEL_MEANS)

# SET ROOT DIRECTORY
cfg.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# FORM ADDRESS TO THE DATA DIRECTORY
cfg.DATA_DIR = osp.join(cfg.ROOT_DIR, cfg.DATA_DIR)

try:
    assert os.environ['DEBUG'] == '1'
    cfg.DEBUG = True
except:
    cfg.DEBUG = False


def get_output_dir(imdb_name, net_name=None, output_dir='output', idx=-1):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """

    outdir = osp.abspath(
        osp.join(cfg.ROOT_DIR, output_dir, cfg.EXP_DIR, imdb_name))
    if net_name is not None:
        outdir = osp.join(outdir, net_name)
    if idx >= 0:
        outdir = osp.join(outdir, str(idx))

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def cfg_print(cfg):
    print('\x1b[32m\x1b[1m' + '#' * 20 + ' Configuration Begins ' + '#' * 20 +
          '\x1b[0m')
    print(toml.dumps(_sort_dict(cfg)))
    print('\x1b[32m\x1b[1m' + '#' * 20 + ' Configuration Ends ' + '#' * 20 +
          '\x1b[0m')


def cfg_dump(cfg, file):
    toml.dump(_sort_dict(cfg), file)


def cfg_table(cfg):
    table = "|key|value|\n|---|---|\n"
    raw_txt = toml.dumps(_sort_dict(cfg)).split('\n')
    for raw_line in raw_txt:
        line = raw_line.split('=')
        if len(line) == 0:
            continue
        if len(line) == 1 and len(line[0]) > 0:
            table += "|**{}**||\n".format(line[0])
        elif len(line) == 2:
            table += "|{}|{}|\n".format(line[0], line[1])
    return table


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not EasyDict:
        return

    for k, v in a.iteritems():
        # do not merge LOG
        if k == "LOG":
            continue
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            # handle unicode
            elif (isinstance(b[k], str) or isinstance(b[k], unicode)) and (
                    isinstance(v, str) or isinstance(v, unicode)):
                pass
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(
                                      type(b[k]), type(v), k))

        # recursively merge dicts
        if type(v) is EasyDict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    amend_config = EasyDict(toml.load(filename))
    _merge_a_into_b(amend_config, cfg)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = cfg
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey), 'Please put {} in default.toml'.format(
            subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        d[subkey] = value
