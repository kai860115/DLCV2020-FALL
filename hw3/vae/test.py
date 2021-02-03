import os
import argparse
import glob
from dataset import myDataset
import torch
from torch.backends import cudnn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision
from model import VAE
from PIL import Image
import numpy as np
import random


def main(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    ])

    model = VAE().cuda()

    state = torch.load(config.ckp_path)
    model.load_state_dict(state['state_dict'])

    os.makedirs(os.path.dirname(config.save_path), exist_ok=True)

    torch.manual_seed(66666)
    np.random.seed(66666)
    random.seed(66666) 
    z = torch.randn((32, 512)).cuda()
    predict = model.decode(z)
    torchvision.utils.save_image(predict.data, config.save_path, nrow=8, normalize=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Testing configuration.
    parser.add_argument('--save_path', type=str, default='./fig1_4.png')
    parser.add_argument('--ckp_path', default='ckpt/vae_5e-6_relu/62500-vae.pth', type=str, help='Checkpoint path.')
    

    config = parser.parse_args()
    print(config)
    main(config)
