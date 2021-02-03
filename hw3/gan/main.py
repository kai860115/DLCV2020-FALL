import os
import argparse
from solver import Solver
from dataset import myDataset
from torch.backends import cudnn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


def main(config):
    # For fast training.
    cudnn.benchmark = True

    if config.use_wandb:
        import wandb
        wandb.init(project="dlcv-hw3-2", config=config)
        #wandb.config.update(vars(args))
        config = wandb.config
        print(config)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    ])
    trainset = myDataset(root=config.data_path, transform=transform)
    trainset_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=1)
    
    solver = Solver(trainset_loader, config)
    solver.train()
        
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # path.
    parser.add_argument('--data_path', type=str, default='../hw3_data/face/train', help="the path of the dataset to train.")
    parser.add_argument('--ckp_dir', default='ckpt/', type=str, help='Checkpoint path', required=False)
    parser.add_argument('--name', default='', type=str, help='Name for saving model')

    # Model configuration.
    parser.add_argument('--nc', type=int, default=3, help='number of channels in the training images')
    parser.add_argument('--nz', type=int, default=100, help='length of latent vector')
    parser.add_argument('--ngf', type=int, default=64, help='relates to the depth of feature maps carried through the generator')
    parser.add_argument('--ndf', type=int, default=64, help='sets the depth of feature maps propagated through the discriminator')

    # training configuration.
    parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of total epoch')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument("--g_lr", help="the learning rate for the generator", default=2e-4 , type=float)
    parser.add_argument("--d_lr", help="the learning rate for the discriminator", default=2e-4 , type=float)
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')

    # Step size.
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--save_interval', type=int, default=250)

    # Others
    parser.add_argument("--use_wandb", help="log training with wandb, "
        "requires wandb, install with \"pip install wandb\"", action="store_true")
    

    config = parser.parse_args()
    print(config)
    main(config)
