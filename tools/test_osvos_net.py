#!/usr/bin/env python

# --------------------------------------------------------
# Multitask Network Cascade
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

# Standard module
import argparse
import sys
import os
import time
import pprint
# User-defined module
import _init_paths
import caffe
from mnc_config import cfg, cfg_from_file
from db.imdb import get_imdb
from caffeWrapper.TesterWrapper import TesterWrapper

# OSVOS - finetune on one-shot frame and then test segmentation on the rest of
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--task', dest='task_name',
                        help='set task name', default='sds',
                        type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    while not os.path.exists(args.caffemodel) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel))
        time.sleep(10)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)


    # Todo - Gilad: train over data augmentation DAVIS
    ### go over all the sequences in DAVIS
    # create a osvos file from aumentations finetune_osvos.txt
    # for finetuning osvos :  args.imdb_name = 'davis_2016_seg_osvos'
    # init the SolverWrapper with imdb
    # train_net on one shot augmented data
    # create new testing file seq_val.txt
    # for testing sequence : args.imdb_name  = 'davis_2016-seg_valSequence
    # save the model, and load it in TesterWrapper


    imdb = get_imdb(args.imdb_name)
    _tester = TesterWrapper(args.prototxt, imdb, args.caffemodel, args.task_name)
    _tester.get_result()
