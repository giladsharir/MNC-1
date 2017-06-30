import glob

import numpy as np
import os
import os.path as osp
from PIL import Image
from enum import Enum
from tqdm import tqdm

from gygox.config import davis_sequences
from gygox.helper_functions.helper_functions import cd


class AugmentationType(Enum):
    NONE = 0
    LOW = 20
    MEDIUM = 38
    HIGH = 83
    EXTREME = 227
    INSANE = 632


def augment_frame_and_gt(save_path, gt_path, data_path,
                         augmentation_type=AugmentationType.LOW,
                         save_files=True, return_in_memory_images=False,
                         empahsize_original_image_by_repeating=True,
                         verbose=False):
    """
    Creates augmetation images for both the input frame and ground truth.
    The amount of the created augmentations would be according to the input augmentation_type.
        :param save_path:
        :param gt_path:
        :param data_path:
        :param augmentation_type:
        :param verbose:
        :return final_filenames: array of paths to augmented filenames, for both images and gt
                                (will be empty if save_files=False)
        :return final_images: array of augmented images, for both images and gt
                                (will be empty if save_files=False)
    """

    aug_filenames = {}
    aug_images = {}
    paths = [gt_path, data_path]
    types = ['gt', 'images']
    aug_count = 0
    no_aug_save_path = ""

    if augmentation_type == AugmentationType.NONE:
        scales = zip([1], ['scale_1'])
        angles = zip([0], ['_0'])
        mirrors = zip([0], ['_0'])
        rolls_x = zip([0], ['_x0'])
        rolls_y = zip([0], ['_y0'])
    elif augmentation_type == AugmentationType.LOW:
        scales = zip([1, 0.95, 1.05], ['scale_1', 'scale_0.95', 'scale_1.05'])
        angles = zip([0, 5, 355], ['_0', '_5', '_355'])
        mirrors = zip([0, 1], ['_0', '_1'])
        rolls_x = zip([0], ['_x0'])
        rolls_y = zip([0], ['_y0'])
    elif augmentation_type == AugmentationType.MEDIUM:
        scales = zip([1, 0.95, 1.05], ['scale_1', 'scale_0.95', 'scale_1.05'])
        angles = zip([0, 5, 355], ['_0', '_5', '_355'])
        mirrors = zip([0], ['_0'])
        rolls_x = zip([0, -0.2], ['_x0', '_x-0.2'])
        rolls_y = zip([0, -0.2], ['_y0', '_y-0.2'])
    elif augmentation_type == AugmentationType.HIGH:
        scales = zip([1, 0.95, 1.05], ['scale_1', 'scale_0.95', 'scale_1.05'])
        angles = zip([0, 5, 355], ['_0', '_5', '_355'])
        mirrors = zip([0], ['_0'])
        rolls_x = zip([0, -0.2, +0.2], ['_x0', '_x-0.2', '_x+0.2'])
        rolls_y = zip([0, -0.2, +0.2], ['_y0', '_y-0.2', '_y+0.2'])
    elif augmentation_type == AugmentationType.EXTREME:
        scales = zip([1, 0.9, 0.8, 1.1, 1.2],
                     ['scale_1', 'scale_0.9', 'scale_0.8', 'scale_1.1',
                      'scale_1.2'])
        angles = zip([0, 5, 10, 355, 350], ['_0', '_5', '_10', '_355', '_350'])
        mirrors = zip([0], ['_0'])
        rolls_x = zip([0, -0.2, +0.2], ['_x0', '_x-0.2', '_x+0.2'])
        rolls_y = zip([0, -0.2, +0.2], ['_y0', '_y-0.2', '_y+0.2'])
    elif augmentation_type == AugmentationType.INSANE:
        scales = zip([1, 0.9, 0.8, 1.1, 1.2],
                     ['scale_1', 'scale_0.9', 'scale_0.8', 'scale_1.1',
                      'scale_1.2'])
        angles = zip([0, 5, 10, 355, 350, 90, 270],
                     ['_0', '_5', '_10', '_355', '_350', '_90', '_270'])
        mirrors = zip([0, 1], ['_0', '_1'])
        rolls_x = zip([0, -0.2, +0.2], ['_x0', '_x-0.2', '_x+0.2'])
        rolls_y = zip([0, -0.2, +0.2], ['_y0', '_y-0.2', '_y+0.2'])

    for p, ptype in zip(paths, types):
        aug_filenames[ptype] = []
        aug_images[ptype] = []

        IM1 = Image.open(p)
        if ptype == 'gt':
            IM1 = IM1.split()[0]

        imname = p.split('/')[-1]
        for r, r_str in scales:
            save_path1 = save_path + '/' + ptype + '/' + r_str
            IM2 = IM1.resize(size=(int(IM1.width * r), int(IM1.height * r)))

            for ang, a_str in angles:
                for i, i_str in mirrors:
                    for roll_x, roll_x_str in rolls_x:
                        for roll_y, roll_y_str in rolls_y:
                            save_path2 = save_path1 + a_str + i_str + roll_x_str + roll_y_str + '/'
                            IM3 = IM2.rotate(ang)
                            IM3 = np.asarray(IM3, dtype=np.uint8)

                            IM3 = np.roll(IM3,
                                          int(round(roll_y * IM3.shape[0], 0)),
                                          axis=0)
                            IM3 = np.roll(IM3,
                                          int(round(roll_x * IM3.shape[0], 0)),
                                          axis=1)

                            if i == 1:
                                IM3 = IM3[:, ::-1]
                            #h = IM3.shape[0]
                            #w = IM3.shape[1]

                            aug_count += 1;

                            if save_files:
                                IM = Image.fromarray(IM3)

                                if not os.path.isdir(save_path2):
                                    os.makedirs(save_path2)
                                # print ("saving %d: ")%(aug_count) + os.path.join(save_path2,imname)
                                IM.save(os.path.join(save_path2, imname))
                                aug_filenames[ptype].append(
                                    os.path.join(save_path2, imname))
                                # print ("saved %d: ")%(aug_count) + os.path.join(save_path2,imname)
                            if return_in_memory_images:
                                aug_images[ptype].append(IM3)

    final_filenames = {}
    final_images = {}

    if empahsize_original_image_by_repeating:
        final_filenames[types[0]] = []
        final_filenames[types[1]] = []
        if save_files:
            final_filenames[types[0]].append(aug_filenames[types[0]][0])
            final_filenames[types[1]].append(aug_filenames[types[1]][0])
            final_filenames[types[0]].extend(aug_filenames[types[0]])
            final_filenames[types[0]].append(final_filenames[types[0]][0])
            final_filenames[types[1]].extend(aug_filenames[types[1]])
            final_filenames[types[1]].append(final_filenames[types[1]][0])
        final_images[types[0]] = []
        final_images[types[1]] = []
        if return_in_memory_images:
            final_images[types[0]].append(aug_images[types[0]][0])
            final_images[types[1]].append(aug_images[types[1]][0])
            final_images[types[0]].extend(aug_images[types[0]])
            final_images[types[0]].append(final_images[types[0]][0])
            final_images[types[1]].extend(aug_images[types[1]])
            final_images[types[1]].append(final_images[types[1]][0])
    else:
        final_filenames = aug_filenames
        final_images = aug_images


    if verbose:
        print "finished aug yay! over all made %d augmentations, and returned a list of size %d." % (
            aug_count, max(len(final_filenames[types[0]]),len(final_images[types[0]])))


    return final_filenames, final_images


def augment_images(davis_db_path):
    # todo: make this function more robust and pythonic
    data_path = osp.join(davis_db_path, 'JPEGImages/480p')
    gt_path = osp.join(davis_db_path, 'Annotations/480p')
    aug_path = 'Augmentation/480p'

    # get images in path
    save_path = aug_path
    paths = [gt_path, data_path]
    types = ['gt', 'images']
    for p, ptype in zip(paths, types):
        print p
        for q, vid in enumerate(tqdm(glob.glob(p + "/*"))):
            vname = vid.split('/')[-1]

            for t, im in enumerate(glob.glob(vid + "/*")):
                IM1 = Image.open(im)
                imname = im.split('/')[-1]

                for r, r_str in zip([1, 0.5, 1.5],
                                    ['scale_1', 'scale_0.5', 'scale_1.5']):
                    #             for r,r_str in zip([0.7,0.5,0.8],['scale_1','scale_0.5','scale_1.5']):
                    save_path1 = osp.join(save_path, ptype, r_str)

                    IM2 = IM1.resize(
                        size=(int(IM1.width * r), int(IM1.height * r)))

                    for ang, a_str in zip([0, 10, 350],
                                          ['_0', '_10', '_350']):
                        for i, i_str in zip([0, 1], ['_0', '_1']):
                            save_path2 = save_path1 + a_str + i_str + '/'
                            IM3 = IM2.rotate(ang)
                            IM3 = np.asarray(IM3, dtype=np.uint8)

                            if i == 1:
                                IM3 = IM3[:, ::-1]
                            h = IM3.shape[0]
                            w = IM3.shape[1]
                            #                         IM3 = IM3[int(h/20):-int(h/20),int(w/20):-int(w/20)]
                            IM = Image.fromarray(IM3)

                            if not osp.isdir(save_path2):
                                os.mkdir(save_path2)
                            IM.save(
                                osp.join(save_path2,
                                         vname + '_' + imname))


def create_data_source_txtfile(
        aug_folder='/home/ubuntu/Data/DAVIS/Augmentation'):
    # todo: make this function more robust and pythonic
    # first create the files:
    # ```all_img.txt and all_gt.txt ```
    # containing all the frame and gt paths (training and testing)
    with cd(aug_folder):
        os.system(
            'for i in JPEGImages/480p/*; do for j in $i/*; do echo $j >> all_frames.txt; done; done')
        os.system(
            'for i in Annotations/480p/*; do for j in $i/*; do echo $j >> all_gt.txt; done; done')

    test_vids = davis_sequences.davis['val_sequences']

    with \
            open(osp.join(aug_folder, 'all_frames.txt')) as p1, \
            open(osp.join(aug_folder, 'all_gt.txt')) as p2, \
            open(osp.join(aug_folder, 'training_pair.lst'), 'w') as p3, \
            open(osp.join(aug_folder, 'testing_pair.lst'), 'w') as p4:
        for l1, l2 in zip(p1.readlines(), p2.readlines()):
            temp = l1.strip().split('/')[-1]
            name = temp[:temp.find('_')]
            if name in test_vids:
                temp2 = l1.strip().split('/')[-2]
                flip = temp2.split('_')[-1]
                rot = temp2.split('_')[-2]
                scale = temp2.split('_')[-3]
                if scale == '1' and rot == '0' and flip == '0':
                    p4.write(l1.strip() + ' ' + l2.strip() + '\n')
            else:
                p3.write(l1.strip() + ' ' + l2.strip() + '\n')


def augment_davis(davis_db_path='/home/ubuntu/Data/DAVIS'):
    """
    Create augmentations on the DAVIS dataset

    Runtime on DAVIS 50 sequences: ~40 minutes

    :return:
    """
    augment_images(davis_db_path)
    create_data_source_txtfile(osp.join(davis_db_path, 'Augmentation'))


if __name__ == '__main__':
    print('running')
    augment_davis()
