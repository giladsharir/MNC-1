import sys

import matplotlib.pyplot as plt
import numpy as np
import os

from gygox.config.config import cfg

sys.path.insert(0, os.path.join(cfg['paths']['caffe_root'], 'python'))
import caffe


class CaffeNet():
    """
    Any DNN done in caffe framework
    """

    def __init__(self, solver_path, net_path='', gpu_on=True, verbose=False):
        if verbose:
            print('Was caffe installed with python layers enabled?')
            print(('Python' in caffe.layer_type_list()))

        self.paths = {
            'solver': solver_path,
            'caffe_root': cfg['paths']['caffe_root'],
            'net': net_path
        }
        self.solver = None
        if gpu_on:
            caffe.set_mode_gpu()
            caffe.set_device(0)

        self.verbose = verbose

    def draw_net2(self, out_path='net_graph.png', rankdir='TB'):
        pass
        # command example: $ caffe/python/draw_net.py <netprototxt_filename> <out_img_filename>
        # import sys
        # sys.path.insert(0, self.paths['caffe_root'] + 'python')
        # caffe.draw.draw_net_to_file(self.paths['net'], out_path, rankdir)

        # command = os.path.join(self.paths['caffe_root'],
        #                        'python/draw_net.py') + ' ' +\
        #           self.paths['net'] + ' ' + out_path
        # os.system(command)

    @staticmethod
    def vis_square(data):
        """Take an array of shape (n, height, width) or (n, 3, height, width)
           and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
        if len(data.shape) == 4:
            data = data.transpose(0, 2, 3, 1)

        # normalize data for display
        data = (data - data.min()) / (data.max() - data.min())

        # force the number of filters to be square
        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = (((0, n ** 2 - data.shape[0]),
                    (0, 1), (0, 1))  # add some space between filters
                   + ((0, 0),) * (
                       data.ndim - 3))  # don't pad the last dimension (if there is one)
        data = np.pad(data, padding, mode='constant',
                      constant_values=1)  # pad with ones (white)

        # tile the filters into an image
        data = data.reshape((n, n) + data.shape[1:]).transpose(
            (0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
        data = data.reshape(
            (n * data.shape[1], n * data.shape[3]) + data.shape[4:])

        plt.imshow(data)
        plt.axis('off')
