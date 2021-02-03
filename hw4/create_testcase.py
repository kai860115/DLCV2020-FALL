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

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class NShotTaskSampler(Sampler):
    def __init__(self, csv_path, episodes_per_epoch, N_way, N_shot, N_query):
        self.data_df = pd.read_csv(csv_path)
        self.N_way = N_way
        self.N_shot = N_shot
        self.N_query = N_query
        self.episodes_per_epoch = episodes_per_epoch

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            batch = []
            episode_classes = np.random.choice(self.data_df['label'].unique(), size=self.N_way, replace=False)

            support = []
            query = []

            for k in episode_classes:
                ind = self.data_df[self.data_df['label'] == k]['id'].sample(self.N_shot + self.N_query).values
                support = support + list(ind[:self.N_shot])
                query = query + list(ind[self.N_shot:])

            batch = support + query

            yield np.stack(batch)

    def __len__(self):
        return self.episodes_per_epoch

def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")

    # training configuration.
    parser.add_argument('--episodes_per_epoch', default=600, type=int, help='episodes per epoch')
    parser.add_argument('--N_way', default=5, type=int, help='N_way (default: 5)')
    parser.add_argument('--N_shot', default=1, type=int, help='N_shot (default: 1)')
    parser.add_argument('--N_query', default=15, type=int, help='N_query (default: 15)')
    parser.add_argument('--test_csv', default='./hw4_data/val.csv', type=str, help="Testing images csv file")
    parser.add_argument('--test_data_dir', type=str, default='./hw4_data/val', help="Testing images directory")

    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()

    batch_sampler = NShotTaskSampler(args.test_csv, args.episodes_per_epoch, args.N_way, args.N_shot, args.N_query)

    line = 'episode_id'
    for i in range(args.N_way):
        for j in range(args.N_shot):
            line += ',class%d_support%d' % (i, j)
    for i in range(args.N_query * args.N_way):
        line += ',query%d' % (i)
    
    print(line)

    for i, data in enumerate(batch_sampler):
        line = '%d' % (i)
        for j in data:
            line += ',%d' % j
        print(line)
