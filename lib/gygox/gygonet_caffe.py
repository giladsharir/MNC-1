import os
import time

import numpy as np
from PIL import Image
from tqdm import tqdm

from config import davis_sequences, gygo_sequences
from caffenet.caffenet import CaffeNet
import caffe
from helper_functions import augmentations
from helper_functions import bilateralSolver as bs
from helper_functions.augmentations import AugmentationType
from helper_functions.helper_functions import caffe_interp_surgery, \
    run_davis_benchmark
from helper_functions.plot_functions import convert_maps_to_masks

from db.imdb import get_imdb
from db.roidb import attach_roidb
from db.maskdb import attach_maskdb
from caffeWrapper.SolverWrapper import SolverWrapper
from caffeWrapper.TesterWrapper import TesterWrapper
from mnc_config import get_output_dir

class GygoNet(CaffeNet):
    """
    A CNN that segments an object in a video

    Currently based on OneShot Video Object Segmentation
    """

    def __init__(self, solver_path, base_model_path='', db_path='',
                 db_name='davis',
                 gt_path='', finetune_pair_path='', testing_pair_path='',
                 net_path='', os_aug_temp_path='', imdb_name='', gpu_on=True,
                 tqdm_off=False):
        """
        :param solver_path:
        :param base_model_path:
        :param db_path:
        :param db_name: Choose one of ['davis', 'gygo']
        :param gt_path:
        :param finetune_pair_path:
        :param net_path:
        :param deploy_path: (currently not in use)
        :param os_aug_temp_path: where to save the temp OS augmentations
        :param gpu_on: bool if to run on GPU
        """
        CaffeNet.__init__(self, solver_path, net_path)
        self.db_name = db_name
        self.paths.update({
            'base_model': base_model_path,
            'db': db_path,
            'gt': gt_path,
            'os_aug_temp': os_aug_temp_path,
            # The following two must be same as in the net.prototxt
            'finetune_pair': finetune_pair_path,
            'testing_pair': testing_pair_path
        })
        if self.paths['db']:
            if not self.paths['gt']:
                if self.db_name == 'davis':
                    self.paths['gt'] = os.path.join(self.paths['db'],
                                                    'Annotations', '480p')

                elif self.db_name == 'gygo':
                    self.paths['gt'] = os.path.join(self.paths['db'],
                                                    'Annotations')

            if not self.paths['finetune_pair']:
                self.paths['finetune_pair'] = os.path.join(
                    self.paths['db'], 'Augmentation',
                    'finetune_os_pair.lst')

            if not self.paths['os_aug_temp']:
                self.paths['os_aug_temp'] = os.path.join(self.paths['db'],
                                                         'Temp', '480p')

            # These are all the sequences in DAVIS
            self.davis = davis_sequences.davis
            self.gygo = gygo_sequences.gygo

        if gpu_on:
            caffe.set_mode_gpu()
            caffe.set_device(0)

        if os_aug_temp_path and not os.path.exists(os_aug_temp_path):
            os.makedirs(os_aug_temp_path)

        self.tqdm_off = tqdm_off

        #Gilad - modified from train_net.py MNC
        self.imdb_name = imdb_name


    def finetune_os(self, os_frame_path, os_gt_path, solver_steps=20,
                    needs_deconv_surgery=False,
                    os_aug_type=AugmentationType.NONE,
                    solver=None):
        """
        finetunes gygox net according to the given 1shot frame and ground truth

        Inputs:
        - os_frame_path:
        - os_gt_path:
        - needs_deconv_surgery: bool if to load separaetly the deconv params.
            Needed when the model doesn't include them.
        - solver_steps: controls how long (#iterations) the solver runs
        - os_aug_type: If to use OS augmentations and what type

        Uses:
        - solver_path:
        - finetune_pair:
        - base_model:

        Returns:
        - solver: finetuned caffe solver instance
        """
        # fine tune on the 1shot image
        if self.verbose:
            print "finetuning on one shot.."
            print os_frame_path
            print os_gt_path
            time.sleep(0.1)

        force_solver_steps_to_fit_os_aug_type = True
        if (force_solver_steps_to_fit_os_aug_type) and (
                    os_aug_type != AugmentationType.NONE):
            solver_steps = os_aug_type.value

        if os_aug_type == AugmentationType.NONE:
            with open(self.paths['finetune_pair'], 'w') as fos:
                fos.write(os_frame_path + ' ' + os_gt_path + '\n')
        else:
            if self.verbose:
                print "creating augmentations.."
            finetune_files, aug_images = augmentations.augment_frame_and_gt(
                self.paths['os_aug_temp'], os_gt_path, os_frame_path,
                augmentation_type=os_aug_type)
            with open(self.paths['finetune_pair'], "w") as fos:
                for ft_img, ft_gt in zip(finetune_files['images'],
                                         finetune_files['gt']):
                    fos.write(ft_img + ' ' + ft_gt + '\n')

            if self.verbose:
                print "finished preparing one shot augmentation and saving to lst file."

        #Gilad - modified from train_net.py (MNC)
        mdb, roidb = attach_roidb(self.imdb_name)
        imdb, maskdb = attach_maskdb(self.imdb_name)
        output_dir = get_output_dir(imdb, None)

        print 'Output will be saved to `{:s}`'.format(output_dir)

        _solver = SolverWrapper(self.paths['solver'], roidb, maskdb, output_dir, imdb,
                                pretrained_model=self.paths['base_model'])

        print 'Solving one-shot fine-tune...'
        finetuned_model_name = _solver.train_model(solver_steps)
        #args.max_iters
        print 'done solving'


        #Gilad - solver not needed for MNC training
        # if not solver:
        #     if self.verbose:
        #         print 'solver path: ' + self.paths['solver']
        #     solver = caffe.SGDSolver(self.paths['solver'])
        #
        #     if self.verbose:
        #         time.sleep(0.1)
        #         print 'loaded solver'
        #
        #     if needs_deconv_surgery:
        #         # do net surgery to set the deconvolution weights for bilinear interpolation
        #         interp_layers = [k for k in solver.net.params.keys() if
        #                          'up' in k]
        #         caffe_interp_surgery(solver.net, interp_layers)
        #         if self.verbose:
        #             print 'interp_surgery done'
        #             time.sleep(0.1)
        #
        #     # copy base weights for fine-tuning
        #     solver.net.copy_from(self.paths['base_model'])
        #     if self.verbose:
        #         time.sleep(0.1)
        #
        # # fine-tune net on 1shot frame
        # for i in tqdm(range(solver_steps), 'finetuning', disable=self.tqdm_off):
        #     solver.step(1)

        return finetuned_model_name

    def object_extract_one_frame(self, fname, map_name, net):
        """
        performs object extraction on a single frame using the given net

        Inputs:
        - fname: file path to the image we want to work on
        - map_name: file path to where to save output map
        - net: the net to use for the task

        Returns:
        """
        if self.verbose:
            print fname
        I = Image.open(fname)
        r = 1
        im = I.resize(size=(int(I.width * r), int(I.height * r)))
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:, :, ::-1]
        in_ -= np.array((104.00698793, 116.66876762, 122.67891434))
        im_shape = np.asarray(im).shape[:2]
        in_ = in_.transpose((2, 0, 1))

        net.blobs['data'].reshape(1, *in_.shape)
        net.blobs['data'].data[...] = in_

        net.forward()

        # out1 = net.blobs['sigmoid-dsn1'].data[0][0, :, :]
        # out2 = net.blobs['sigmoid-dsn2'].data[0][0, :, :]
        # out3 = net.blobs['sigmoid-dsn3'].data[0][0, :, :]
        # out4 = net.blobs['sigmoid-dsn4'].data[0][0, :, :]
        # out5 = net.blobs['sigmoid-dsn5'].data[0][0, :, :]
        fuse = net.blobs['sigmoid-fuse'].data[0][0, :, :]

        # scale_lst = [out1, out2, out3, out4, out5]

        # Optional: apply bilateral solver
        grid = bs.BilateralGrid(np.asarray(im), **bs.grid_params)

        im_shape = (im.height, im.width)
        target = fuse
        confidence = np.ones(shape=target.shape)  # *(2e16-1)
        t = target.reshape(-1, 1).astype(np.double)  # / (pow(2,16)-1)
        c = confidence.reshape(-1, 1).astype(np.double)  # / (pow(2,16)-1)

        output_solver = bs.BilateralSolver(
            grid, bs.bs_params).solve(t, c).reshape(im_shape)

        # Save output image
        if not os.path.exists(os.path.dirname(map_name)):
            os.makedirs(os.path.dirname(map_name))
        out_image = Image.fromarray(output_solver * 255).convert('RGB')
        out_image.save(map_name)
        if self.verbose:
            print "output path: " + map_name

        return net

    def run_on_seq(self, frames_list, os_frame_path, os_gt_path, maps_list,
                   os_aug_type=AugmentationType.NONE, solver_steps=20, sequence_name = ''):
        """
        fine-tune gygox net with 1shot and run segmentation a single video sequence

        :param frames_list: list of frame paths to be segmented
        :param os_frame_path: path of the one-shot fine-tuning frame
        :param os_gt_path: path of the one-shot fine-tuning ground-truth
        :param maps_list: list of frame paths to write the outputs to
        :param os_aug_type: If to use OS augmentations and what type
        :param solver_steps: controls how long (#iterations) the solver runs
        :return:

        """
        # prepare the testing_pair.lst file.
        # The second part of each line doesn't matter
        with open(self.paths['testing_pair'], "w") as fos:
            for ft_img, irrelevant in zip(frames_list, frames_list):
                fos.write(' '.join([ft_img, irrelevant + '\n']))

        # fine-tune the net
        finetune_model_name = self.finetune_os(
            os_frame_path, os_gt_path, solver_steps, os_aug_type=os_aug_type)

        print "testing finetuned model {}".format(finetune_model_name)
        #new imdb for testing frames
        imdb = get_imdb(self.imdb_name+"_test")
        _tester = TesterWrapper(self.paths['net'], imdb, finetune_model_name, 'seg')
        _tester.get_result(seq_name=sequence_name)
        # recreate the net
        # net = self._instance_net_from_solver(solver)
        # del solver

        # run on the rest of the frames and save results
        # for fname, map_name in tqdm(zip(frames_list, maps_list),
        #                             desc='Segmenting frames',
        #                             disable=self.tqdm_off):
        #     self.object_extract_one_frame(fname, map_name, net)
        # del net

    def run_on_dataset(self, chosen_sequences=[], os_solver_steps=20,
                       os_aug_type=AugmentationType.NONE,
                       os_train_frame_idx='first', maps_out_path='',
                       masks_out_path=''):
        """
        runs a net on the DAVIS dataset.

        :param chosen_sequences:
        :param os_solver_steps:
        :param os_train_frame_idx: 'first'/'middle'
        :param maps_out_path: where to save the segmentation output in grayscale level
        :param masks_out_path: where to save the binarized segmentation output
        :param db: ['davis', 'gygo'] - swtiches between the db to run on
        :return:
        """
        db_path = self.paths['db']

        if not maps_out_path:
            maps_out_path = os.path.join(
                db_path, 'Results', 'Maps', '480p',
                'vl-middle-frame-maps')
        if not masks_out_path:
            masks_out_path = os.path.join(
                db_path, 'Results', 'Segmentations', '480p',
                'vl-middle-frame')

        if not os.path.exists(maps_out_path):
            os.makedirs(maps_out_path)

        if not os.path.exists(masks_out_path):
            os.makedirs(masks_out_path)

        if not chosen_sequences:
            if self.db_name == 'davis':
                chosen_sequences = self.davis['val_sequences']
            elif self.db_name == 'gygo':
                chosen_sequences = self.gygo['val_sequences']
        # File containing validation frame pairs (image and annotation)
        if self.db_name == 'davis':
            all_val_pair_paths = os.path.join(db_path, 'ImageSets', '480p',
                                              'val.txt')
        elif self.db_name == 'gygo':
            all_val_pair_paths = os.path.join(db_path, 'ImageSets', 'val.txt')

        # measuring time of main loop
        start_time = time.time()

        for seq_number, seq in enumerate(chosen_sequences):
            print ("%d:%s:") % (seq_number, seq)
            time.sleep(0.1)

            # Build frame_paths and maps_paths:
            frame_paths = []
            maps_paths = []
            with open(all_val_pair_paths) as val_pair_file:
                for line in val_pair_file.readlines():
                    if self.db_name == 'davis':
                        # sample line:
                        # '/JPEGImages/480p/bmx-trees/00072.jpg /Annotations/480p/bmx-trees/00072.png '
                        line_seq = line.split('/')[3]
                    elif self.db_name == 'gygo':
                        # sample line:
                        # /Frames/seg_prod_t01/0x1TFTRZ5NGJ/00000.jpg /Annotations/seg_prod_t01/0x1TFTRZ5NGJ/00000.png
                        line_seq = '/'.join(line.split('/')[2:4])
                    # Only keep lines relating to the current sequence
                    if (line_seq != seq):
                        continue
                    split = line.split(' ')[:2]
                    frame_paths.append(
                        os.path.join(db_path, split[0][1:]))
                    if self.db_name == 'davis':
                        maps_paths.append(split[1].split('480p')[1][1:])
                    if self.db_name == 'gygo':
                        maps_paths.append(
                            split[1].split('Annotations')[1][1:-1])

            # Choose the 1shot frame(s)
            if os_train_frame_idx == 'first':
                os_frame_idx = 0
            elif os_train_frame_idx == 'middle':
                os_frame_idx = len(frame_paths) / 2
            os_frame_path = frame_paths[os_frame_idx]
            os_gt_path = os.path.join(self.paths['gt'],
                                      maps_paths[os_frame_idx])

            # finalize full maps path
            maps_paths = [os.path.join(maps_out_path, path) for path in
                          maps_paths]

            self.run_on_seq(frame_paths, os_frame_path, os_gt_path, maps_paths,
                            os_aug_type=os_aug_type,
                            solver_steps=os_solver_steps,
                            sequence_name=seq)

        #@Gilad - modified form gygonet - not needed since MNC tester saves masks
        # convert_maps_to_masks(
        #     maps_path=maps_out_path, save_to=masks_out_path, threshold=178.5)

        print "--- total time for main loop: %s seconds ---" % (
            time.time() - start_time)

    def train(self, train_steps_nb, interp_surgery_on=True):
        """
        train on davis data

        :param train_steps_nb:
        :return:
        """
        if self.verbose:
            print('Creating solver...')
        self.solver = caffe.SGDSolver(self.paths['solver'])
        if self.verbose:
            print('Performing interpolation surgery...')
        if interp_surgery_on:
            # do net surgery to set the deconvolution weights for bilinear interpolation
            interp_layers = [k for k in self.solver.net.params.keys() if
                             'up' in k]
            caffe_interp_surgery(self.solver.net, interp_layers)

        if self.paths['base_model']:
            if self.verbose:
                print('Copying base model...')
            self.solver.net.copy_from(self.paths['base_model'])

        for i in tqdm(range(train_steps_nb), disable=self.tqdm_off):
            self.solver.step(1)

        if self.verbose:
            print('Training done!')

    def _instance_net_from_solver(self, solver, net_path=''):
        if not net_path:
            net_path = self.paths['net']
        net = caffe.Net(net_path, caffe.TEST)

        params = net.params.keys()
        target_params = {pr: (net.params[pr][0].data, net.params[pr][1].data)
                         for pr in params}
        source_params = {
            pr: (solver.net.params[pr][0].data, solver.net.params[pr][1].data)
            for
            pr in params}
        for pr in params:
            target_params[pr][1][...] = source_params[pr][1]  # bias
            target_params[pr][0][...] = source_params[pr][0]  # weights

        if self.verbose:
            print 'loaded net'
        return net

    def _instance_solver_with_copy_params_from_solver(self, solver_old):

        solver = caffe.SGDSolver(self.paths['solver_path'])

        params = solver_old.net.params.keys()
        target_params = {
        pr: (solver.net.params[pr][0].data, solver.net.params[pr][1].data)
        for pr in params}
        source_params = {
            pr: (solver_old.net.params[pr][0].data,
                 solver_old.net.params[pr][1].data)
            for
            pr in params}
        for pr in params:
            target_params[pr][1][...] = source_params[pr][1]  # bias
            target_params[pr][0][...] = source_params[pr][0]  # weights

        return solver

    def eval_davis_benchmark(self, davis_benchmark_dir, masks_dir, eval_set):
        run_davis_benchmark(davis_benchmark_dir,
                            davis_db_dir=self.paths['db'],
                            masks_dir=masks_dir, eval_set=eval_set)

    @classmethod
    def display_side_outputs(cls, net,
                             blobs_list=['upscore-dsn1', 'upscore-dsn2',
                                         'upscore-dsn3'
                                 , 'upscore-dsn4', 'upscore-dsn5',
                                         'upscore-fuse']
                             , save_fig_path='', show_fig=False):
        import matplotlib.pyplot as plt

        myblobs = np.array(
            [net.blobs[blobs_list[x]].data for x in range(len(blobs_list))])

        print myblobs.shape

        maxs = [np.max(myblobs[x, 0, 0, :, :]) for x in range(len(myblobs))]
        print maxs
        mins = [np.min(myblobs[x, 0, 0, :, :]) for x in range(len(myblobs))]
        print mins

        blob_max = np.max(maxs)
        blob_min = np.min(mins)
        print blob_max
        print blob_min

        edge = max(blob_max, -blob_min)

        f, axarr = plt.subplots(len(blobs_list), 3,
                                figsize=(14, 2 + 2 * len(blobs_list)))
        for x in range(len(myblobs)):
            axarr[x, 0].imshow(myblobs[x, 0, 0, :, :])
            axarr[x, 0].axis('off')
            axarr[x, 1].imshow(myblobs[x, 0, 0, :, :], cmap="RdBu", vmin=-edge,
                               vmax=edge)
            axarr[x, 1].axis('off')
            axarr[x, 2].imshow(myblobs[x, 0, 0, :, :], vmin=0, vmax=1)
            axarr[x, 2].axis('off')

        if save_fig_path:
            fig1 = plt.gcf()
        if show_fig:
            plt.show()
        if save_fig_path:
            fig1.savefig(save_fig_path)
