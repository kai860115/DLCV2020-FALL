import os
import argparse
import glob
from dataset import myDataset
import torch
from torch.backends import cudnn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision
from model import Generator
from PIL import Image
import numpy as np
import random


def main(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    ])
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    model = Generator(nc=config.nc, ngf=config.ngf, nz=config.nz).cuda()

    state = torch.load(config.ckp_path)
    model.load_state_dict(state)

    os.makedirs(os.path.dirname(config.save_path), exist_ok=True)

    seed = 222
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed) 
    z = torch.randn((32, config.nz, 1, 1)).cuda()
    predict = model(z)
    torchvision.utils.save_image(predict.data, config.save_path, nrow=8, normalize=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Testing configuration.
    parser.add_argument('--save_path', type=str, default='./fig2_2.png')
    parser.add_argument('--ckp_path', default='ckpt/gan/31250-G.pth', type=str, help='Checkpoint path.')
    parser.add_argument('--nc', type=int, default=3, help='number of channels in the training images')
    parser.add_argument('--nz', type=int, default=100, help='length of latent vector')
    parser.add_argument('--ngf', type=int, default=64, help='relates to the depth of feature maps carried through the generator')
    

    config = parser.parse_args()
    print(config)
    main(config)
