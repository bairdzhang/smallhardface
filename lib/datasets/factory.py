from datasets.wider import wider
from datasets.fddb import fddb
from datasets.general import general
from datasets.pascalface import pascalface
from datasets.afw import afw

__sets = {}

for split in ['train', 'val', 'test']:
    name = 'wider_{}'.format(split)
    __sets[name] = (lambda split=split: wider(split))

for split in ['val']:
    name = 'fddb_{}'.format(split)
    __sets[name] = (lambda split=split: fddb(split))

for split in ['png', 'jpg']:
    name = 'general_{}'.format(split)
    __sets[name] = (lambda split=split: general(split))

for split in ['val']:
    name = 'pascalface_{}'.format(split)
    __sets[name] = (lambda split=split: pascalface(split))

for split in ['val']:
    name = 'afw_{}'.format(split)
    __sets[name] = (lambda split=split: afw(split))


def get_imdb(name, path=None):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()
