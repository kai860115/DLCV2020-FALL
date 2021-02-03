import re
import glob
import os
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image

def mask_target(im):
    im = transforms.ToTensor()(im)
    im = 4 * im[0] + 2 * im[1] + 1 * im[2]
    target = torch.zeros(im.shape, dtype=torch.long)
    target[im==3] = 0
    target[im==6] = 1
    target[im==5] = 2
    target[im==2] = 3
    target[im==1] = 4
    target[im==7] = 5
    target[im==0] = 6
    target[im==4] = 6
            
    return target

class myDataset(Dataset):
    def __init__(self, root, transform=None, randomflip=False):
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.transform = transform
        self.randomflip = randomflip

        # read filenames
        sat_filenames = glob.glob(os.path.join(root, '*.jpg'))
        sat_filenames.sort()
        mask_filenames = glob.glob(os.path.join(root, '*.png'))
        mask_filenames.sort()

        for sat_fn, mask_fn in zip(sat_filenames, mask_filenames):
            self.filenames.append((sat_fn, mask_fn)) 
                
        self.len = len(self.filenames)
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        sat_fn, mask_fn = self.filenames[index]
        sat = Image.open(sat_fn)
        mask = Image.open(mask_fn)

        if (self.randomflip):
            if random.random() > 0.5:
                sat = TF.hflip(sat)
                mask = TF.hflip(mask)

            if random.random() > 0.5:
                sat = TF.vflip(sat)
                mask = TF.vflip(mask)
            
        if self.transform is not None:
            sat = self.transform(sat)

        return sat, mask_target(mask)

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len
