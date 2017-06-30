import glob
import os

import cv2

if __name__ == "__main__":
    """
    Convert the binary ouput images to [0..255] images that can be viewed
    """
    # MAKE CHANGES HERE
    folder_prefix = '/Volumes/Public/GyGO-object-results/2016-12-28-osvos_Iter15000'
    sequence_names = ['paragliding-launch',
                      'parkour',
                      'kite-surf',
                      'breakdance',
                      'dog',
                      'drift-chicane',
                      'libby',
                      'goat',
                      'car-roundabout',
                      'car-shadow',
                      'bmx-trees',
                      'blackswan',
                      'dance-twirl',
                      'motocross-jump',
                      'cows',
                      'camel',
                      'horsejump-high',
                      'drift-straight',
                      'scooter-black',
                      'soapbox']
    outdir_prefix = 'Visible'

    # DON'T CHANGE BELOW
    folder_names = [os.path.join(folder_prefix, str) for str in sequence_names]
    for ind, dir in enumerate(folder_names):
        dir_name = os.path.join(outdir_prefix, sequence_names[ind])
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        image_paths = sorted(glob.glob(os.path.join(dir, '*')))
        for im_path in image_paths:
            im = cv2.imread(im_path)
            im = im * 255
            out_path = os.path.join(dir_name, os.path.basename(im_path))
            cv2.imwrite(out_path, im)
