# -*- coding: utf-8 -*-
'''
    options for training/test script.

    Author : NoUnique (kofmap@gmail.com)
    Copyright 2020 NoUnique. All Rights Reserved
'''

from absl import flags

## flag for flagfile
#flags.DEFINE_string('flagfile', './configs/train-cifar100_001.conf',
#                    'Insert flag definitions from the given file into the command line.')

# Training settings
flags.DEFINE_string('DATASET_DIR', default='/data',
                    help='Directory where dataset exists.')
flags.DEFINE_string('TENSORBOARD_DIR', default='./checkpoints',
                    help='Directory where tensorboard scalars and graph are written to.')
flags.DEFINE_string('CHECKPOINT_DIR', default='./checkpoints',
                    help='Directory where checkpoints and event logs are written to.')
flags.DEFINE_string('CHECKPOINT_FORMAT', default='checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
flags.DEFINE_string('PRETRAINED_CHECKPOINT_PATH', default=None,
                    help='checkpoint file path to load')
flags.DEFINE_boolean('FP16_ALLREDUCE', default=False,
                     help='use fp16 compression during allreduce')
flags.DEFINE_integer('BATCHES_PER_ALLREDUCE', default=1,
                     help='number of batches processed locally before '
                          'executing allreduce across workers; it multiplies '
                          'total batch size.')
flags.DEFINE_boolean('USE_ADASUM', default=False,
                     help='use adasum algorithm to do reduction')
flags.DEFINE_integer('LOG_INTERVAL', default=10,
                     help='how many batches to wait before logging training status')

# Default settings from https://arxiv.org/abs/1706.02677.
flags.DEFINE_string('MODEL_NAME', 'MobileNetV2',
                    help='The name of the architecture to train.')
flags.DEFINE_string('DATASET_NAME', 'CIFAR100',
                    help='The name of the dataset to train.')
flags.DEFINE_multi_float('DATA_MEAN', [0.5071, 0.4867, 0.4408],
                         help='mean value of dataset')
flags.DEFINE_multi_float('DATA_STD', [0.2675, 0.2565, 0.2761],
                         help='standard deviation value of dataset')
flags.DEFINE_multi_integer('DATA_SHAPE', [3, 32, 32],
                           help='data dimension of dataset')
flags.DEFINE_list('BLOCK_ARGS', ['wm1.0_rn8_s1',
                                 't1_c16_n1_s1',
                                 't6_c24_n2_s1',
                                 't6_c32_n3_s2',
                                 't6_c64_n4_s2',
                                 't6_c96_n3_s1',
                                 't6_c160_n3_s2',
                                 't6_c320_n1_s1'],
                  help='argument of blocks in EfficientNet style')
flags.DEFINE_integer('BATCH_SIZE', default=128,
                     help='input batch size for training')
flags.DEFINE_integer('VALID_BATCH_SIZE', default=32,
                     help='input batch size for validation')
flags.DEFINE_integer('TEST_BATCH_SIZE', default=1,
                     help='input batch size for test')
flags.DEFINE_integer('EPOCHS', default=90,
                     help='number of epochs to train')
flags.DEFINE_float('BASE_LR', default=0.0125,
                   help='learning rate for a single GPU')
flags.DEFINE_integer('WARMUP_EPOCHS', default=5,
                     help='number of warmup epochs')
flags.DEFINE_float('WARMUP_START_LR', default=0.0001,
                   help='learning rate for initial warmup epoch')
flags.DEFINE_string('LR_POLICY', default='cosine',
                    help='Learning rate policy')
flags.DEFINE_multi_integer('LR_DECAY_EPOCHS', [0, 20, 30],
                           help='Decay epochs for step_decay lr scheduler')
flags.DEFINE_multi_float('LR_DECAY_LRS', [1, 0.1, 0.001],
                         help='Decay epochs for step_decay lr scheduler')
flags.DEFINE_float('LR_DECAY_POWER', default=0.9,
                   help='power value for polynomial_decay lr scheduler')
flags.DEFINE_float('MOMENTUM', default=0.9,
                   help='SGD momentum')
flags.DEFINE_float('WEIGHT_DECAY', default=0.0001,
                   help='weight decay')
flags.DEFINE_boolean('NESTEROV', default=True,
                     help='use Nesterov SGD')

flags.DEFINE_boolean('CUDA', default=True,
                     help='enables CUDA training')
flags.DEFINE_integer('NUM_THREADS', default=4,
                     help='number of threads for cpu ops')
flags.DEFINE_integer('NUM_WORKERS', default=4,
                     help='number of worker for dataloader')
flags.DEFINE_integer('SEED', default=42,
                     help='random seed')
flags.DEFINE_boolean('EVAL_WITH_TESTSET', False,
                     help='Use test set when evaluating trained model')

FLAGS = flags.FLAGS


def flags_to_string(flag_list):
    dictionary = flag_list.flag_values_dict()
    string = '\nFLAGS\n'
    for key in dictionary.keys():
        if key.isupper():
            string += '    {}: {}\n'.format(key, dictionary[key])
    return string