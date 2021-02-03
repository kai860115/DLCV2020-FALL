import glob
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class myDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.img_paths = []
        for fn in glob.glob(os.path.join(root, '*.png')):
            self.img_paths.append(fn)

        self.len = len(self.img_paths)


    def __getitem__(self, index):
        """ Get a sample from the dataset """
        fn = self.img_paths[index]
        img = Image.open(fn)
        if self.transform is not None:
            img = self.transform(img)

        return img, img

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len