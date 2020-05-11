# -*- coding: utf-8 -*-
"""
    learning rate schedulers.

    Author : NoUnique (kofmap@gmail.com)
    Copyright 2020 NoUnique. All Rights Reserved
"""

import math
import horovod.torch as hvd


# Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
def adjust_learning_rate(FLAGS, optimizer, cur_epoch):
    lr = get_lr_at_epoch(FLAGS, cur_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * hvd.size() * FLAGS.BATCHES_PER_ALLREDUCE
    return lr * hvd.size() * FLAGS.BATCHES_PER_ALLREDUCE


def get_lr_at_epoch(FLAGS, cur_epoch):
    lr = get_lr_func(FLAGS.LR_POLICY)(FLAGS, cur_epoch)
    # Perform warm up.
    if cur_epoch < FLAGS.WARMUP_EPOCHS:
        lr_start = FLAGS.WARMUP_START_LR
        lr_end = get_lr_func(FLAGS.LR_POLICY)(
            FLAGS, FLAGS.WARMUP_EPOCHS
        )
        alpha = (lr_end - lr_start) / FLAGS.WARMUP_EPOCHS
        lr = cur_epoch * alpha + lr_start
    return lr


def lr_func_cosine(FLAGS, cur_epoch):
    return (
        FLAGS.BASE_LR
        * (math.cos(math.pi * cur_epoch / FLAGS.EPOCHS) + 1.0)
        * 0.5
    )


def lr_func_steps_with_relative_lrs(FLAGS, cur_epoch):
    ind = get_step_index(FLAGS, cur_epoch)
    return FLAGS.LR_DECAY_LRS[ind] * FLAGS.BASE_LR


def get_step_index(FLAGS, cur_epoch):
    steps = FLAGS.LR_DECAY_EPOCHS + [FLAGS.EPOCHS]
    index = None
    for ind, step in enumerate(steps):  # NoQA
        index = ind
        if cur_epoch < step:
            break
    return index - 1


def lr_func_poly(FLAGS, cur_epoch):
    return round(
        FLAGS.BASE_LR * (1 - cur_epoch / FLAGS.EPOCHS) ** FLAGS.LR_DECAY_POWER,
        ndigits=8
    )


def get_lr_func(lr_policy):
    policy = "lr_func_" + lr_policy
    if policy not in globals():
        raise NotImplementedError("Unknown LR policy: {}".format(lr_policy))
    else:
        return globals()[policy]