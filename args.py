#!/usr/bin/env python
#-*- coding:utf-8 _*-

import argparse


TRAIN_DATASETS = ['laplace2d_simple', 'laplace2d_mixed_simple', 'darcy2d_simple',
                  'heat2d_simple', 'nonlinear_poisson2d_simple']

INFERENCE_DATASETS = ['laplace2d_schwarz', 'laplace2d_holes', 'laplace2d_bosch', 'laplace2d_dolphin', 'laplace2d_disk',
                      'laplace2d_mixed_schwarz', 'laplace2d_mixed_holes', 'laplace2d_mixed_bosch',
                      'darcy2d_schwarz', 'darcy2d_holes', 'darcy2d_bosch', 'darcy2d_negative_triangle',
                      'heat2d_schwarz', 'heat2d_holes', 'heat2d_bosch',
                      'nonlinear_poisson2d_schwarz', 'nonlinear_poisson2d_holes', 'nonlinear_poisson2d_bosch']


def _add_common_args(parser):
    """Add arguments shared between training and inference."""
    parser.add_argument('--component', type=str, default='all')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--use-tb', type=int, default=0, help='whether use tensorboard')
    parser.add_argument('--comment', type=str, default="", help="comment for the experiment")
    parser.add_argument('--train-num', type=str, default='all')
    parser.add_argument('--test-num', type=str, default='all')
    parser.add_argument('--sort-data', type=int, default=0)
    parser.add_argument('--normalize_x', type=str, default='unit',
                        choices=['none', 'minmax', 'unit'])
    parser.add_argument('--normalize_y', type=str, default='unit',
                        choices=['none', 'minmax', 'unit', 'quantile'],
                        help="whether normalize y")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--loss-name', type=str, default='rel2',
                        choices=['rel2', 'rel1', 'l2', 'l1'])

    # model architecture
    parser.add_argument('--model-name', type=str, default='GNOT',
                        choices=['CGPT', 'GNOT'])
    parser.add_argument('--n-hidden', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=3)

    # MLP / attention
    parser.add_argument('--act', type=str, default='gelu',
                        choices=['gelu', 'relu', 'tanh', 'sigmoid'])
    parser.add_argument('--n-head', type=int, default=1)
    parser.add_argument('--ffn-dropout', type=float, default=0.0,
                        help='dropout for the FFN in attention')
    parser.add_argument('--attn-dropout', type=float, default=0.0)
    parser.add_argument('--mlp-layers', type=int, default=3)
    parser.add_argument('--attn-type', type=str, default='linear',
                        choices=['random', 'linear', 'gated', 'hydra', 'kernel'])
    parser.add_argument('--hfourier-dim', type=int, default=0)

    # GNOT
    parser.add_argument('--n-experts', type=int, default=1)
    parser.add_argument('--branch-sizes', nargs="*", type=int, default=[2])
    parser.add_argument('--n-inner', type=int, default=4)

    # time dependent
    parser.add_argument('--time-dependent', type=bool, default=False)
    parser.add_argument('--time-step', type=int, default=1)

    return parser


def get_train_parser():
    parser = argparse.ArgumentParser(description='GNOT for operator learning')
    parser.add_argument('--dataset', type=str, default='ns2d',
                        choices=TRAIN_DATASETS)
    _add_common_args(parser)

    # training-specific
    parser.add_argument('--seed', type=int, default=2023, metavar='Seed',
                        help='random seed (default: 2023)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['Adam', 'AdamW'])
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='max learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=5e-6)
    parser.add_argument('--grad-clip', type=str, default=1000.0)
    parser.add_argument('--batch-size', type=int, default=4, metavar='bsz',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--val-batch-size', type=int, default=8, metavar='bsz',
                        help='input batch size for validation (default: 8)')
    parser.add_argument('--lr-method', type=str, default='cycle',
                        choices=['cycle', 'step', 'warmup'])
    parser.add_argument('--lr-step-size', type=int, default=50)
    parser.add_argument('--warmup-epochs', type=int, default=50)
    parser.add_argument('--resume', type=str, default=None,
                        help='path to checkpoint to resume training from')
    return parser


def get_train_args():
    return get_train_parser().parse_args()


def get_inference_parser():
    parser = argparse.ArgumentParser(description='inference with SNI')
    parser.add_argument('--dataset', type=str, default='laplace2d_schwarz',
                        choices=INFERENCE_DATASETS)
    _add_common_args(parser)

    parser.add_argument('--model-path', type=str)

    # domain decomposition
    parser.add_argument('--n-parts', type=int, default=10)
    parser.add_argument('--tau', type=float)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=5000)

    # time dependent (inference-specific overrides)
    parser.add_argument('--time-span', type=int, default=1)
    # override --time-step default from 1 (training) to 5 (inference)
    parser.set_defaults(time_step=5)

    return parser


def get_inference_args():
    return get_inference_parser().parse_args()