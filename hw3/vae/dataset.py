import glob
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class myDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.images = []
        for fn in glob.glob(os.path.join(root, '*.png')):
            img = Image.open(fn)
            if self.transform is not None:
                img = self.transform(img)
            
            self.images.append(img)

        self.len = len(self.images)


    def __getitem__(self, index):
        """ Get a sample from the dataset """
        img = self.images[index]

        return img, img

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len