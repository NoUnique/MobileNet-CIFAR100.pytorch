# -*- coding: utf-8 -*-
"""
    training script which trains a model using a given dataset.

    Author : NoUnique (kofmap@gmail.com)
    Copyright 2020 NoUnique. All Rights Reserved
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import os
import tqdm
import argparse
import datetime
import importlib

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.tensorboard as tensorboard
import torch.utils.data.distributed as distributed
import numpy as np
import horovod.torch as hvd

from absl import app

from lib.metrics import Metric, accuracy
from lib.utils import get_real_path, ssl_set_unverified_context
from lib.utils import compute_inference_time, calculate_model_complexity
from options import FLAGS, flags_to_string


def main(_):
    """ Basic Configurations """
    ssl_set_unverified_context()
    FLAGS.CUDA = FLAGS.CUDA and torch.cuda.is_available()
    hvd.init()
    np.random.seed(FLAGS.SEED)
    torch.manual_seed(FLAGS.SEED)

    if FLAGS.CUDA:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(FLAGS.SEED)

    cudnn.benchmark = True

    # Horovod: print logs on the first worker.
    verbose = 1 if hvd.rank() == 0 else 0

    # Select subdirectory as datetime if flagfile is not specified
    subdir = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    # If sys.argv has flagfile argument, set subdir as filename of flagfile
    parser = argparse.ArgumentParser()
    parser.add_argument('--flagfile')
    for flag in FLAGS.flag_values_dict().keys():
        if flag.isupper():
            parser.add_argument('--' + flag)
    args = parser.parse_args()
    if args.flagfile is not None:
        flagfile = args.flagfile
        subdir = os.path.splitext(os.path.basename(flagfile))[0]
        subdir = os.path.join(subdir, '-'.join(FLAGS.BLOCK_ARGS))


    """ Prepare Dataset """
    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(FLAGS.NUM_THREADS)

    kwargs = {'num_workers': FLAGS.NUM_WORKERS, 'pin_memory': True} if FLAGS.CUDA else {}

    dataset_module = 'lib.data.datasets.' + FLAGS.DATASET_NAME.lower()
    dataset = importlib.import_module(dataset_module).__getattribute__(FLAGS.DATASET_NAME)
    test_dataset = dataset('test', data_dir=FLAGS.DATASET_DIR)

    # Horovod: use DistributedSampler to partition data among workers. Manually specify
    # `num_replicas=hvd.size()` and `rank=hvd.rank()`.
    test_sampler = distributed.DistributedSampler(test_dataset,
                                                  num_replicas=hvd.size(),
                                                  rank=hvd.rank())
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=FLAGS.TEST_BATCH_SIZE,
                                              sampler=test_sampler, **kwargs)


    """ Build Model """
    # Set up a model.
    model_module = 'models.' + FLAGS.MODEL_NAME.lower()
    net = importlib.import_module(model_module).__getattribute__(FLAGS.MODEL_NAME)
    model = net(num_classes=len(test_dataset.classes), block_args=FLAGS.BLOCK_ARGS)

    # By default, Adasum doesn't need scaling up learning rate.
    # For sum/average with gradient Accumulation: scale learning rate by batches_per_allreduce

    if FLAGS.CUDA:
        # Move model to GPU.
        model.cuda()


    """ Restore & Broadcast """
    # If set > 0, will resume training from a given checkpoint.
    resume_from_epoch = 0
    fileroot = get_real_path(FLAGS.CHECKPOINT_DIR)
    for try_epoch in range(FLAGS.EPOCHS, 0, -1):
        filename = FLAGS.CHECKPOINT_FORMAT.format(epoch=try_epoch)
        filepath = os.path.join(fileroot, subdir, filename)
        if os.path.exists(filepath):
            resume_from_epoch = try_epoch
            break

    # Horovod: broadcast resume_from_epoch from rank 0 (which will have
    # checkpoints) to other ranks.
    resume_from_epoch = hvd.broadcast(torch.tensor(resume_from_epoch), root_rank=0,
                                      name='resume_from_epoch').item()

    # Restore from a previous checkpoint, if initial_epoch is specified.
    # Horovod: restore on the first worker which will broadcast weights to other workers.
    if FLAGS.PRETRAINED_CHECKPOINT_PATH is not None:
        filepath = get_real_path(FLAGS.PRETRAINED_CHECKPOINT_PATH)
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model'])
        print('load weight: {}'.format(filepath))
    elif resume_from_epoch > 0 and hvd.rank() == 0:
        fileroot = get_real_path(FLAGS.CHECKPOINT_DIR)
        filename = FLAGS.CHECKPOINT_FORMAT.format(epoch=resume_from_epoch)
        filepath = os.path.join(fileroot, subdir, filename)
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model'])
        print('load weight: {}'.format(filepath))

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    def test():
        model.eval()
        valid_loss = Metric('valid_loss')
        valid_accuracy = Metric('valid_accuracy')

        with tqdm.tqdm(total=len(test_loader),
                       desc='Test Model',
                       disable=not verbose) as t:
            with torch.no_grad():
                for data, target in test_loader:
                    if FLAGS.CUDA:
                        data, target = data.cuda(), target.cuda()
                    output = model(data)

                    valid_loss.update(F.cross_entropy(output, target))
                    valid_accuracy.update(accuracy(output, target))
                    t.set_postfix({'loss': valid_loss.avg.item(),
                                   'accuracy': 100. * valid_accuracy.avg.item()})
                    t.update(1)
        print("test result: {:.2f}".format(valid_accuracy.avg * 100))


    """ Training Loop """
    if hvd.rank() == 0:
        print(model)
        calculate_model_complexity(model, input_dim=FLAGS.DATA_SHAPE, cuda=FLAGS.CUDA)
        compute_inference_time(model, input_dim=FLAGS.DATA_SHAPE, cuda=FLAGS.CUDA)
        print(flags_to_string(FLAGS))
    test()


if __name__ == '__main__':
    app.run(main)