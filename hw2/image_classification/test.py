import os
import argparse
import glob
from solver import Solver
from dataset import myDataset
import torch
from torch.backends import cudnn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from model import Model
from PIL import Image


def main(config):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    model = Model().to(device)

    state = torch.load(config.ckp_path)
    model.load_state_dict(state['state_dict'])

    filenames = glob.glob(os.path.join(config.img_dir, '*.png'))
    filenames = sorted(filenames)
    out_filename = os.path.join(config.save_dir, 'test_pred.csv')
    os.makedirs(config.save_dir, exist_ok=True)
    model.eval()
    with open(out_filename, 'w') as out_file:
        out_file.write('image_id,label\n')
        with torch.no_grad():
            for fn in filenames:
                data = Image.open(fn)
                data = transform(data)
                data = torch.unsqueeze(data, 0)
                data = data.to(device)
                _, output = model(data)
                pred = output.max(1, keepdim=True)[1]
                out_file.write(fn.split('/')[-1] + ',' + str(pred.item()) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training configuration.
    parser.add_argument('--img_dir', type=str, default='../hw2_data/p1_data/val_50')
    parser.add_argument('--save_dir', type=str, default='ckpt/vgg16_bn')
    parser.add_argument('--ckp_path', default='../hw2_1_model.pkl', type=str, help='Checkpoint path.')
    
    config = parser.parse_args()
    print(config)
    main(config)
