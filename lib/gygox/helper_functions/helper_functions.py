import numpy as np
import os
import yaml


def caffe_interp_surgery(net, layers):
    """
    set parameters s.t. deconvolutional layers compute bilinear interpolation

    N.B. this is for deconvolution without groups
    """

    def upsample_filt(size):
        """
        make a bilinear interpolation kernel

        credit @longjon
        """
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        return (1 - abs(og[0] - center) / factor) * \
               (1 - abs(og[1] - center) / factor)

    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k:
            print 'input + output channels need to be the same'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt


class cd:
    """Context manager for changing the current working directory"""

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def run_davis_benchmark(davis_benchmark_dir, davis_db_dir, masks_dir,
                        eval_set):
    """
    Runs the davis benchmark on the given segmented sequences

    Sample terminal lines to reproduce in this function:
        cd ~/Dev/davis/python
        export PYTHONPATH=$(pwd)/lib
        workon davis # activate virtualenv
        python tools/eval.py data/DAVIS/Results/Segmentations/480/osvos data/DAVIS/Results/Evaluations/480p
        python eval_view.py --eval_set=test davis/data/DAVIS/Results/Evaluations/480p/osvos.h5

    :davis_benchmark_dir: location of the davis repo
    :param davis_db_dir: root of davis_db
    :param masks_dir: folder with segmented sequences to be evaluated
    :eval_set: which set should be evaluated.
        Choose from: ['test', 'train', 'gygo-test', 'gygo-train']
    :return:
    """
    # Todo: this function is currently not working

    eval_dir = os.path.join(davis_db_dir, 'Results', 'Evaluations')
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    eval_file = os.path.join(eval_dir, os.path.basename(masks_dir) + '.h5')
    davis_benchmark_python_dir = os.path.join(davis_benchmark_dir, 'python')

    with cd(davis_benchmark_python_dir):
        os.system('export PYTHONPATH=$(pwd)/lib')
        print('Activating davis python environment...')
        os.system('workon davis')
        print('Preparig evaluation...')
        os.system(' '.join(['python tools/eval.py', masks_dir, eval_dir]))
        print('Displaying evaluation...')
        os.system(' '.join(['python eval_view.py', '--eval_set=' + eval_set,
                            eval_file]))
