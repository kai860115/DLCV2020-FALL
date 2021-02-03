import os
import argparse
import glob
from dataset import myDataset
import torch
from torch.backends import cudnn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from model import DANN
from PIL import Image


def main(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    ])
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    model = DANN().cuda()

    state = torch.load(config.ckp_path)
    model.load_state_dict(state['state_dict'])

    filenames = glob.glob(os.path.join(config.img_dir, '*.png'))
    filenames = sorted(filenames)

    out_filename = config.save_path
    os.makedirs(os.path.dirname(config.save_path), exist_ok=True)

    model.eval()
    with open(out_filename, 'w') as out_file:
        out_file.write('image_name,label\n')
        with torch.no_grad():
            for fn in filenames:
                data = Image.open(fn).convert('RGB')
                data = transform(data)
                data = torch.unsqueeze(data, 0)
                data = data.cuda()
                output, _ = model(data, 1)
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                out_file.write(fn.split('/')[-1] + ',' + str(pred.item()) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training configuration.
    parser.add_argument('--img_dir', type=str, default='../hw3_data/digits/svhn/test')
    parser.add_argument('--save_path', type=str, default='ckpt/test')
    parser.add_argument('--ckp_path', default='ckpt/test/3750-dann.pth', type=str, help='Checkpoint path.')
    

    config = parser.parse_args()
    print(config)
    main(config)
