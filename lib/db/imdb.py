# --------------------------------------------------------
# Multitask Network Cascade
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from datasets.pascal_voc_det import PascalVOCDet
from datasets.pascal_voc_seg import PascalVOCSeg
from datasets.davis_seg import DAVISSeg

# __sets = {
#     'voc_2012_seg_train': (lambda: PascalVOCSeg('train', '2012', 'data/VOCdevkitSDS/')),
#     'voc_2012_seg_val': (lambda: PascalVOCSeg('val', '2012', 'data/VOCdevkitSDS/')),
#     'voc_2007_trainval': (lambda: PascalVOCDet('trainval', '2007')),
#     'voc_2007_test': (lambda: PascalVOCDet('test', '2007'))
# }
__sets = {
    'voc_2012_seg_train': (lambda: PascalVOCSeg('train', '2012', '/home/ubuntu/Dev/VOCdevkitSDS/')),
    'voc_2012_seg_val': (lambda: PascalVOCSeg('val', '2012', '/home/ubuntu/Dev/VOCdevkitSDS/')),
    'voc_2007_trainval': (lambda: PascalVOCDet('trainval', '2007')),
    'voc_2007_test': (lambda: PascalVOCDet('test', '2007')),
    'davis_2016_seg_train': (lambda: DAVISSeg('train', '2016', '/home/ubuntu/Data/DAVIS/')),
    'davis_2016_seg_val': (lambda: DAVISSeg('val', '2016', '/home/ubuntu/Data/DAVIS/')),
    'davis_2016_seg_osvos': (lambda: DAVISSeg('finetune_osvos', '2016', '/home/ubuntu/Data/DAVIS/', finetune_mode=True, use_cache=False)),
    'davis_2016_seg_osvos_test': (lambda: DAVISSeg('test_osvos', '2016', '/home/ubuntu/Data/DAVIS/', use_cache=False))
}


def get_imdb(name):
    """ Get an imdb (image database) by name.
    """
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()


def list_imdbs():
    return __sets.keys()
