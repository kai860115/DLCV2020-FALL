import os
import argparse
from solver import Solver
from dataset import get_dataset
from torch.backends import cudnn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


def main(config):
    # For fast training.
    cudnn.deterministic = True
    cudnn.benchmark = True

    # load the dataset
    trainset = get_dataset(root=config.trainset_dir, data_aug=config.data_aug)
    valset = get_dataset(root=config.valset_dir, data_aug=config.data_aug)

    # Data loader.
    trainset_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=1)
    valset_loader = DataLoader(valset, batch_size=config.batch_size, shuffle=False, num_workers=1)
    
    solver = Solver(trainset_loader, valset_loader, config)

    solver.train()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training configuration.
    parser.add_argument('--trainset_dir', type=str, default='../hw2_data/p1_data/train_50')
    parser.add_argument('--valset_dir', type=str, default='../hw2_data/p1_data/val_50')
    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size')
    parser.add_argument('--epoch', type=int, default=10, help='number of total epoch')
    parser.add_argument('--resume_iter', type=int, default=0, help='resume training from this iteration')
    parser.add_argument('--name', default='', type=str, help='Name for saving model')
    parser.add_argument('--ckp_dir', default='ckpt/', type=str, help='Checkpoint path', required=False)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=500)
    parser.add_argument("--data_aug", help="data augmentation", action="store_true")
    
    config = parser.parse_args()
    print(config)
    main(config)
