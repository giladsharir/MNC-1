import os
import os.path as osp

from tqdm import tqdm


def create_trainval_txt(frames_dir):
    """
    Creats the trainval.txt file for the gygo dataset

    :param frames_dir: directory where the frames of the dataset are stored
    :return:
    """
    # sample output line:
    # Frames/seg_prod_t01/0x1C3FAIKD8S/00000.jpg Annotations/seg_prod_t01/0x1C3FAIKD8S/00000.png
    n_seq = 122 # number of sequences at the moment
    with open('val.txt', 'w') as valfile:
        for root, dirs, files in tqdm(os.walk(frames_dir), total=n_seq):
            for file in files:
                if file[0] == '.':
                    continue  # ignore hidden file
                relative_root = root[len(frames_dir) + 1:]
                valfile.write(' '.join([
                    osp.join('/Frames', relative_root, file),
                    osp.join('/Annotations', relative_root,
                             osp.splitext(file)[0] + '.png')]) + '\n')


def create_val_txt(frames_dir):
    """
    Creats the val.txt file for the gygo dataset

    :param frames_dir: directory where the frames of the dataset are stored
    :return:
    """
    n_seq = 122 # number of sequences at the moment
    with open('val.txt', 'w') as valfile, open(
            '/Volumes/Public/GyGO-data/datasets/GyGO-Production/ImageSets/gygo-test-set.txt') as test_list:
        for line in test_list.readlines():
            for root, dirs, files in tqdm(os.walk(frames_dir, total=n_seq)):
                for file in files:
                    if file[0] == '.':
                        continue  # ignore hidden file
                    if line[:-1] not in root:
                        continue  # only take files in the val set
                    relative_root = root[len(frames_dir) + 1:]
                    valfile.write(' '.join([
                        osp.join('/Frames', relative_root, file),
                        osp.join('/Annotations', relative_root,
                                 osp.splitext(file)[0] + '.png')]) + '\n')

def create_train_txt(frames_dir):
    """
    Creats the train.txt file for the gygo dataset

    :param frames_dir: directory where the frames of the dataset are stored
    :return:
    """
    n_seq = 122 # number of sequences at the moment
    with open(
        '/Volumes/Public/GyGO-data/datasets/GyGO-Production/ImageSets/gygo-test-set.txt') as test_list:
        lines = test_list.readlines()
    lines = [line[:-1] for line in lines]
    with open('val.txt', 'w') as valfile:
        for root, dirs, files in tqdm(os.walk(frames_dir), total=n_seq):
            for file in files:
                if file[0] == '.':
                    continue  # ignore hidden file
                relative_root = root[len(frames_dir) + 1:]
                if relative_root in lines:
                    continue  # only take files in the val set
                valfile.write(' '.join([
                    osp.join('/Frames', relative_root, file),
                    osp.join('/Annotations', relative_root,
                             osp.splitext(file)[0] + '.png')]) + '\n')

if __name__ == '__main__':
    create_train_txt(
        '/Volumes/Public/GyGO-data/datasets/GyGO-Production/Frames')
