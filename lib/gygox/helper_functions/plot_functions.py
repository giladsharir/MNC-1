import os
import os.path as osp

import cv2
import matplotlib.cm as cm
import matplotlib.pylab as pylab
import numpy as np
import tqdm
from matplotlib import pyplot as plt
from tqdm import tqdm


def plot_single_scale(scale_lst, size):
    pylab.rcParams['figure.figsize'] = size, size / 2

    plt.figure()
    for i in range(0, len(scale_lst)):
        s = plt.subplot(1, 5, i + 1)
        plt.imshow(scale_lst[i], cmap=cm.Greys_r)
        s.set_xticklabels([])
        s.set_yticklabels([])
        s.yaxis.set_ticks_position('none')
        s.xaxis.set_ticks_position('none')
    plt.tight_layout()
    plt.show()


def convert_maps_to_masks(
        maps_path='/home/ubuntu/Data/DAVIS/Augmentation/tmp_result/',
        save_to='/home/ubuntu/Dev/davis/data/DAVIS/Results/Segmentations/480p/osvos/',
        threshold=178.5):
    """
    convert the maps outputted by gygonet to binary segmentation masks

    """
    for root, dirs, files in tqdm(os.walk(maps_path),
                                  'Sequences: Maps-->Masks'):
        for file in files:
            if file[0] == '.':
                continue # ignore hidden files
            relative_root = root[len(maps_path) + 1:]
            prob_map = cv2.imread(osp.join(root, file), 0)
            mask = np.array(prob_map > threshold, dtype=np.uint8) * 255

            dest_dir = osp.join(save_to, relative_root)
            if not osp.exists(dest_dir):
                os.makedirs(dest_dir)
            cv2.imwrite(osp.join(dest_dir, file), mask)

def convert_map_to_mask(map, threshold=178.5, low_val=0, high_val=255):
    """
    Converts map to mask,
    all elements under threshold will get low_val and the rest will get high_val
    in the output mask
    :param map: array with grayscale levels
    :param threshold:
    :param low_val:
    :param high_val:
    :return mask: array with
    """
    mask = np.array(map).copy()
    thresh = threshold / 255.0
    mask_t = (map < thresh)
    mask[mask_t] = low_val
    mask_t = (map >= thresh)
    mask[mask_t] = high_val
    return mask


def display_maps(maps, save_fig_path='',
                 show_fig=False, verbose=False):
    """
    Enables to plot images where each image is plotted three times:
        left: `gray`, middle: `Normalized RdBu`, right: `clipped gray`
        :param maps: array of images of type nparray
        :param save_fig_path: if given saves figure at this path
        :param show_fig: if True show() figure
        :param verbose: if True prints debug info
    """



    if verbose:
        print('len(maps):%d; maps[0].shape:%s' %(len(maps),maps[0].shape.__str__()) )


    maxs = [np.max(maps[x]) for x in range(len(maps))]
    if verbose:
        print maxs
    mins = [np.min(maps[x]) for x in range(len(maps))]
    if verbose:
        print mins

    blob_max = np.max(maxs)
    blob_min = np.min(mins)
    if verbose:
        print blob_max
        print blob_min

    edge = max(blob_max, -blob_min)

    f, axarr = plt.subplots(len(maps), 3,
                            figsize=(14, 2 + 2 * len(maps)))
    if len(maps)>1:
        for x in range(len(maps)):
            axarr[x, 0].imshow(maps[x], cmap="gray")
            axarr[x, 0].axis('off')
            axarr[x, 1].imshow(maps[x], cmap="RdBu", vmin=-edge,
                               vmax=edge)
            axarr[x, 1].axis('off')
            axarr[x, 2].imshow(maps[x], cmap="gray", vmin=0, vmax=1)
            axarr[x, 2].axis('off')
    else:
        axarr[0].imshow(maps[0], cmap="gray")
        axarr[0].axis('off')
        axarr[1].imshow(maps[0], cmap="RdBu", vmin=-edge,
                           vmax=edge)
        axarr[1].axis('off')
        axarr[2].imshow(maps[0], cmap="gray", vmin=0, vmax=1)
        axarr[2].axis('off')

    if save_fig_path:
        fig1 = plt.gcf()
    if show_fig:
        plt.show()
    if save_fig_path:
        fig1.savefig(save_fig_path)

    plt.close()