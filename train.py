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
import sys
import math
import tqdm
import argparse
import datetime
import importlib

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.tensorboard as tensorboard
import torch.utils.data.distributed as distributed
import numpy as np
import horovod.torch as hvd

from absl import app

from lib.data import transforms
from lib.lr_schedules import adjust_learning_rate
from lib.metrics import Metric, accuracy
from lib.utils import get_real_path, ssl_set_unverified_context
from lib.utils import compute_inference_time, calculate_model_complexity
from options import FLAGS, flags_to_string


def main(_):
    """ Basic Configurations """
    ssl_set_unverified_context()
    FLAGS.CUDA = FLAGS.CUDA and torch.cuda.is_available()
    allreduce_batch_size = FLAGS.BATCH_SIZE * FLAGS.BATCHES_PER_ALLREDUCE
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
        script_name = [os.path.splitext(os.path.basename(arg))[0]
                       for arg in sys.argv if arg.endswith('.py')]
        if len(script_name) > 0:
            subdir = subdir.replace('train', script_name[0])

    # Horovod: write TensorBoard logs on first worker.
    if hvd.rank() == 0:
        fileroot = get_real_path(FLAGS.TENSORBOARD_DIR)
        train_tensorboard_dir = os.path.join(fileroot, subdir, 'train')
        valid_tensorboard_dir = os.path.join(fileroot, subdir, 'valid')
        train_summary_writer = tensorboard.SummaryWriter(train_tensorboard_dir)
        valid_summary_writer = tensorboard.SummaryWriter(valid_tensorboard_dir)


    """ Prepare Dataset """
    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(FLAGS.NUM_THREADS)

    kwargs = {'num_workers': FLAGS.NUM_WORKERS, 'pin_memory': True} if FLAGS.CUDA else {}

    dataset_module = 'lib.data.datasets.' + FLAGS.DATASET_NAME.lower()
    dataset = importlib.import_module(dataset_module).__getattribute__(FLAGS.DATASET_NAME)
    train_dataset = dataset('train', data_dir=FLAGS.DATASET_DIR,
                            mean=FLAGS.DATA_MEAN, std=FLAGS.DATA_STD)
    valid_dataset = dataset('valid', data_dir=FLAGS.DATASET_DIR,
                            mean=FLAGS.DATA_MEAN, std=FLAGS.DATA_STD)

    # Horovod: use DistributedSampler to partition data among workers. Manually specify
    # `num_replicas=hvd.size()` and `rank=hvd.rank()`.
    train_sampler = distributed.DistributedSampler(train_dataset,
                                                   num_replicas=hvd.size(),
                                                   rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=allreduce_batch_size,
                                               sampler=train_sampler, **kwargs)

    valid_sampler = distributed.DistributedSampler(valid_dataset,
                                                   num_replicas=hvd.size(),
                                                   rank=hvd.rank())
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=FLAGS.VALID_BATCH_SIZE,
                                               sampler=valid_sampler, **kwargs)


    """ Build Model """
    # Set up a model.
    model_module = 'models.' + FLAGS.MODEL_NAME.lower()
    net = importlib.import_module(model_module).__getattribute__(FLAGS.MODEL_NAME)
    model = net(num_classes=len(train_dataset.classes), block_args=FLAGS.BLOCK_ARGS)

    # By default, Adasum doesn't need scaling up learning rate.
    # For sum/average with gradient Accumulation: scale learning rate by batches_per_allreduce
    lr_scaler = FLAGS.BATCHES_PER_ALLREDUCE * hvd.size() if not FLAGS.USE_ADASUM else 1

    if FLAGS.CUDA:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if FLAGS.USE_ADASUM and hvd.nccl_built():
            lr_scaler = FLAGS.BATCHES_PER_ALLREDUCE * hvd.local_size()


    """ Optimizer """
    # Horovod: scale learning rate by the number of GPUs.
    optimizer = optim.SGD(model.parameters(), lr=(FLAGS.BASE_LR * lr_scaler),
                          momentum=FLAGS.MOMENTUM, weight_decay=FLAGS.WEIGHT_DECAY,
                          nesterov=FLAGS.NESTEROV)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if FLAGS.FP16_ALLREDUCE else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
        compression=compression,
        backward_passes_per_step=FLAGS.BATCHES_PER_ALLREDUCE)
        # TODO: hvd.Adasum is not supported yet(0.18.2)
        #backward_passes_per_step = FLAGS.BATCHES_PER_ALLREDUCE,
        #op = hvd.Adasum if FLAGS.USE_ADASUM else hvd.Average)


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
    if resume_from_epoch > 0 and hvd.rank() == 0:
        fileroot = get_real_path(FLAGS.CHECKPOINT_DIR)
        filename = FLAGS.CHECKPOINT_FORMAT.format(epoch=resume_from_epoch)
        filepath = os.path.join(fileroot, subdir, filename)
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)


    """ Training Operations """
    def train(epoch):
        model.train()
        lr = adjust_learning_rate(FLAGS, optimizer, epoch)
        train_sampler.set_epoch(epoch)
        train_loss = Metric('train_loss')
        train_accuracy = Metric('train_accuracy')

        with tqdm.tqdm(total=len(train_loader),
                       desc='Train Epoch     #{}'.format(epoch + 1),
                       disable=not verbose) as t:
            for batch_idx, (data, target) in enumerate(train_loader):
                if FLAGS.CUDA:
                    data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                # Split data into sub-batches of size batch_size
                for i in range(0, len(data), FLAGS.BATCH_SIZE):
                    data_batch = data[i:i + FLAGS.BATCH_SIZE]
                    target_batch = target[i:i + FLAGS.BATCH_SIZE]
                    output = model(data_batch)
                    train_accuracy.update(accuracy(output, target_batch))
                    loss = F.cross_entropy(output, target_batch)
                    train_loss.update(loss)
                    # Average gradients among sub-batches
                    loss.div_(math.ceil(float(len(data)) / FLAGS.BATCH_SIZE))
                    loss.backward()
                    if i == 0 and hvd.rank() == 0:
                        train_summary_writer.add_image("input",
                                                       transforms.denormalize(data[0],
                                                                              mean=FLAGS.DATA_MEAN,
                                                                              std=FLAGS.DATA_STD),
                                                       epoch)
                # Gradient is applied across all ranks
                optimizer.step()
                t.set_postfix({'loss': train_loss.avg.item(),
                               'accuracy': 100. * train_accuracy.avg.item(),
                               'lr': lr})
                t.update(1)

        if hvd.rank() == 0:
            train_summary_writer.add_scalar('info/lr', lr, epoch)
            train_summary_writer.add_scalar('info/loss', train_loss.avg, epoch)
            train_summary_writer.add_scalar('metric/accuracy', train_accuracy.avg, epoch)

    def validate(epoch):
        model.eval()
        valid_loss = Metric('valid_loss')
        valid_accuracy = Metric('valid_accuracy')

        with tqdm.tqdm(total=len(valid_loader),
                       desc='Validate Epoch  #{}'.format(epoch + 1),
                       disable=not verbose) as t:
            with torch.no_grad():
                for data, target in valid_loader:
                    if FLAGS.CUDA:
                        data, target = data.cuda(), target.cuda()
                    output = model(data)

                    valid_loss.update(F.cross_entropy(output, target))
                    valid_accuracy.update(accuracy(output, target))
                    t.set_postfix({'loss': valid_loss.avg.item(),
                                   'accuracy': 100. * valid_accuracy.avg.item()})
                    t.update(1)

        if hvd.rank() == 0:
            valid_summary_writer.add_scalar('info/loss', valid_loss.avg, epoch)
            valid_summary_writer.add_scalar('metric/accuracy', valid_accuracy.avg, epoch)

    def save_checkpoint(epoch):
        if hvd.rank() == 0:
            fileroot = get_real_path(FLAGS.CHECKPOINT_DIR)
            filename = FLAGS.CHECKPOINT_FORMAT.format(epoch=epoch + 1)
            filepath = os.path.join(fileroot, subdir, filename)
            if not os.path.exists(os.path.dirname(filepath)):
                os.makedirs(os.path.dirname(filepath))
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, filepath)


    """ Training Loop """
    if hvd.rank() == 0:
        print(model)
        print(flags_to_string(FLAGS))
    for epoch in range(resume_from_epoch, FLAGS.EPOCHS):
        train(epoch)
        validate(epoch)
        save_checkpoint(epoch)


if __name__ == '__main__':
    app.run(main)