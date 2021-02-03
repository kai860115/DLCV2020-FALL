import re
import glob
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class myDataset(Dataset):
    def __init__(self, root, transform=None):
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.transform = transform

        # read filenames
        filenames = glob.glob(os.path.join(root, '*.png'))
        for fn in filenames:
            i = int(re.split('/|_', fn)[-2])
            self.filenames.append((fn, i)) # (filename, label) pair
                
        self.len = len(self.filenames)
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn, label = self.filenames[index]
        image = Image.open(image_fn)
            
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

def get_dataset(root, data_aug=False):
    if data_aug:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10, resample=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])

    return myDataset(root, transform)