import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

import csv
import random
import numpy as np
import pandas as pd

from PIL import Image
filenameToPILImage = lambda x: Image.open(x)

from dataset import MiniDataset
from samplers import GeneratorSampler, NShotTaskSampler
from solver import Solver

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def parse_args():
    parser = argparse.ArgumentParser(description="Data Hallucination for Few shot learning")

    # training configuration.
    parser.add_argument('--episodes_per_epoch', default=600, type=int, help='episodes per epoch')
    parser.add_argument('--N_way_train', default=5, type=int, help='N_way (default: 5) for training')
    parser.add_argument('--N_shot_train', default=1, type=int, help='N_shot (default: 1) for training')
    parser.add_argument('--N_query_train', default=15, type=int, help='N_query (default: 15) for training')
    parser.add_argument('--M_aug_train', default=10, type=int, help='M_augmentation (default: 10) for training')
    parser.add_argument('--N_way_val', default=5, type=int, help='N_way (default: 5) for val')
    parser.add_argument('--N_shot_val', default=1, type=int, help='N_shot (default: 1) for val')
    parser.add_argument('--N_query_val', default=15, type=int, help='N_query (default: 15) for val')
    parser.add_argument('--M_aug_val', default=10, type=int, help='M_augmentation (default: 10) for val')
    parser.add_argument('--matching_fn', default='cosine', type=str, help='distance matching function')
    parser.add_argument('--nz', type=int, default=1600, help='length of latent vector')
    parser.add_argument("--data_aug", help="data augmentation", action="store_true")

    # optimizer configuration
    parser.add_argument("--lr", help="the learning rate", default=1e-4, type=float)
    parser.add_argument('--num_steps_decay', type=int, default=40, help='number of steps for decaying lr')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='weight_decay for Adam optimizer')

    # loss configuration
    parser.add_argument('--active_adversarial_loss_step', type=int, default=0, help='number of iterations for activating any additional adversarial loss')
    parser.add_argument('--alpha_weight', type=float, default=0.2, help='weight for adversarial loss')

    # path.
    parser.add_argument('--train_csv', type=str, default='../hw4_data/train.csv', help="Training images csv file")
    parser.add_argument('--train_data_dir', type=str, default='../hw4_data/train', help="Training images directory")
    parser.add_argument('--val_csv', type=str, default='../hw4_data/val.csv', help="val images csv file")
    parser.add_argument('--val_data_dir', type=str, default='../hw4_data/val', help="val images directory")
    parser.add_argument('--val_testcase_csv', type=str, default='../hw4_data/val_testcase.csv', help="val test case csv")
    parser.add_argument('--ckp_dir', default='ckpt/', type=str, help='Checkpoint path', required=False)
    parser.add_argument('--name', default='', type=str, help='Name for saving model')

    # Step size.
    parser.add_argument('--num_epochs', type=int, default=100, help='number of total epochs')
    parser.add_argument('--resume_iter', type=int, default=0, help='resume training from this epoch')
    parser.add_argument('--num_d_steps', type=int, default=1, help='number of training discriminator steps')
    parser.add_argument('--log_interval', type=int, default=300)
    parser.add_argument('--ckp_interval', type=int, default=600)

    # Others
    parser.add_argument("--use_wandb", help="log training with wandb, "
        "requires wandb, install with \"pip install wandb\"", action="store_true")

    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()

    if args.use_wandb:
        import wandb
        wandb.init(project="few-shot-learning", config=args)
        args = wandb.config
        print(args)

    train_dataset = MiniDataset(args.train_csv, args.train_data_dir, args.data_aug)
    val_dataset = MiniDataset(args.val_csv, args.val_data_dir)

    train_loader = DataLoader(
        train_dataset,
        num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
        batch_sampler=NShotTaskSampler(args.train_csv, args.episodes_per_epoch, args.N_way_train, args.N_shot_train, args.N_query_train))

    val_loader = DataLoader(
        val_dataset, batch_size=args.N_way_val * (args.N_query_val + args.N_shot_val),
        num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=GeneratorSampler(args.val_testcase_csv))

    solver = Solver(args, train_loader, val_loader)

    solver.train()