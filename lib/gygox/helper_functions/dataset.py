"""

"""
from PIL import Image
import os
import numpy as np
import sys
import augmentations
from augmentations import AugmentationType


class Dataset:
    def __init__(self, train_list, test_list, database_root, store_memory=True,
                 data_aug=AugmentationType.NONE, tmp_folder='/tmp/gygonet/',
                 random_shuffle_train=True):
        """Initialize the Dataset object
        :param train_list: TXT file or list with the paths of the images to use for training (Images must be between 0 and 255)
        :param test_list: TXT file or list with the paths of the images to use for testing (Images must be between 0 and 255)
        :param database_root: Path to the root of the Database
        :param store_memory: True stores all the training images, False loads at runtime the images
        :param data_aug: What augmentation to perform on the given list(s)
        :param random_shuffle_train: bool if to suffle the train_list
        :param tmp_folder: If `store_memory=False` then where to store the augmentations
        """

        # Load training images (path) and labels
        print('Started loading files...')
        if not isinstance(train_list, list) and train_list is not None:
            with open(train_list) as t:
                train_paths = t.readlines()
        elif isinstance(train_list, list):
            train_paths = train_list
        else:
            train_paths = []
        if not isinstance(test_list, list) and test_list is not None:
            with open(test_list) as t:
                test_paths = t.readlines()
        elif isinstance(test_list, list):
            test_paths = test_list
        else:
            test_paths = []
        self.images_train = []
        self.images_train_path = []
        self.labels_train = []
        self.labels_train_path = []
        self.random_shuffle_train = random_shuffle_train
        for idx, line in enumerate(train_paths):
            os_frame_path   = os.path.join(database_root, str(line.split()[0]).strip("/"))
            os_gt_path      = os.path.join(database_root, str(line.split()[1]).strip("/"))

            if (data_aug != AugmentationType.NONE):

                if idx == 0: sys.stdout.write('Performing the data augmentation\n')

                aug_filepaths, aug_images = augmentations.augment_frame_and_gt(
                                    tmp_folder, os_gt_path, os_frame_path,
                                    augmentation_type=data_aug,
                                    save_files=not(store_memory), return_in_memory_images=store_memory,
                                    empahsize_original_image_by_repeating=True,
                                    verbose=True)
                self.images_train.extend(aug_images['images'])
                self.labels_train.extend(aug_images['gt'])
                self.images_train_path.extend(aug_filepaths['images'])
                self.labels_train_path.extend(aug_filepaths['gt'])
            else:
                if store_memory:
                    img = Image.open(os_frame_path)
                    label = Image.open(os_gt_path).split()[0]
                    if idx == 0: sys.stdout.write('Loading the data')
                    self.images_train.append(np.array(img, dtype=np.uint8))
                    self.labels_train.append(np.array(label, dtype=np.uint8))
                else:
                    self.images_train_path.append(os_frame_path)
                    self.labels_train_path.append(os_gt_path)
            if (idx + 1) % 50 == 0:
                sys.stdout.write('.')

        sys.stdout.write('\n')
        self.images_train_path = np.array(self.images_train_path)
        self.labels_train_path = np.array(self.labels_train_path)

        # Load testing images (path) and labels
        self.images_test = []
        self.images_test_path = []
        for idx, line in enumerate(test_paths):
            if store_memory:
                self.images_test.append(np.array(Image.open(os.path.join(database_root, str(line.split()[0]))),
                                                 dtype=np.uint8))
                if (idx + 1) % 1000 == 0:
                    print('Loaded ' + str(idx) + ' test images')
            self.images_test_path.append(os.path.join(database_root, str(line.split()[0])))
        print('Done initializing Dataset')

        # Init parameters
        self.train_ptr = 0
        self.test_ptr = 0
        self.train_size = max(len(self.images_train_path), len(self.images_train))
        self.test_size = len(self.images_test_path)
        self.train_idx = np.arange(self.train_size)
        if self.random_shuffle_train:
            np.random.shuffle(self.train_idx)
        self.store_memory = store_memory

    def next_batch(self, batch_size, phase):
        """Get next batch of image (path) and labels
        :param batch_size: Size of the batch
        :param phase: Possible options:'train' or 'test'
        Returns in training:
        :return images: List of images paths if store_memory=False, List of Numpy arrays of the images if store_memory=True
        :return labels: List of labels paths if store_memory=False, List of Numpy arrays of the labels if store_memory=True
        Returns in testing:
        :return images: None if store_memory=False, Numpy array of the image if store_memory=True
        :return path: List of image paths
        """
        if phase == 'train':
            if self.train_ptr + batch_size < self.train_size:
                idx = np.array(self.train_idx[self.train_ptr:self.train_ptr + batch_size])
                if self.store_memory:
                    images = [self.images_train[l] for l in idx]
                    labels = [self.labels_train[l] for l in idx]
                else:
                    images = [self.images_train_path[l] for l in idx]
                    labels = [self.labels_train_path[l] for l in idx]
                self.train_ptr += batch_size
            else:
                old_idx = np.array(self.train_idx[self.train_ptr:])
                if self.random_shuffle_train:
                    np.random.shuffle(self.train_idx)
                new_ptr = (self.train_ptr + batch_size) % self.train_size
                idx = np.array(self.train_idx[:new_ptr])
                if self.store_memory:
                    images_1 = [self.images_train[l] for l in old_idx]
                    labels_1 = [self.labels_train[l] for l in old_idx]
                    images_2 = [self.images_train[l] for l in idx]
                    labels_2 = [self.labels_train[l] for l in idx]
                else:
                    images_1 = [self.images_train_path[l] for l in old_idx]
                    labels_1 = [self.labels_train_path[l] for l in old_idx]
                    images_2 = [self.images_train_path[l] for l in idx]
                    labels_2 = [self.labels_train_path[l] for l in idx]
                images = images_1 + images_2
                labels = labels_1 + labels_2
                self.train_ptr = new_ptr
            return images, labels
        elif phase == 'test':
            images = None
            if self.test_ptr + batch_size < self.test_size:
                if self.store_memory:
                    images = self.images_test[self.test_ptr:self.test_ptr + batch_size]
                paths = self.images_test_path[self.test_ptr:self.test_ptr + batch_size]
                self.test_ptr += batch_size
            else:
                new_ptr = (self.test_ptr + batch_size) % self.test_size
                if self.store_memory:
                    images = self.images_test[self.test_ptr:] + self.images_test[:new_ptr]
                paths = self.images_test_path[self.test_ptr:] + self.images_test_path[:new_ptr]
                self.test_ptr = new_ptr
            return images, paths
        else:
            return None, None

    def get_train_size(self):
        return self.train_size

    def get_test_size(self):
        return self.test_size

    def train_img_size(self):
        width, height = Image.open(self.images_train[self.train_ptr]).size
        return height, width

    def reset_iter(self):
        """Resets points to test and train sets,
        such that next_batch() can be called again starting to iterate
        over the dataset from the beginning"""
        self.test_ptr = 0
        self.train_ptr = 0