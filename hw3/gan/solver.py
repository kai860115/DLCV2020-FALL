import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from model import Generator, Discriminator
from utils import weights_init


class Solver(object):
    def __init__(self, trainset_loader, config):
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.trainset_loader = trainset_loader
        self.nc = config.nc
        self.nz = config.nz
        self.ngf = config.ngf
        self.ndf = config.ndf
        self.n_epochs = config.n_epochs
        self.resume_iters = config.resume_iters
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.exp_name = config.name
        os.makedirs(config.ckp_dir, exist_ok=True)
        self.ckp_dir = os.path.join(config.ckp_dir, self.exp_name)
        os.makedirs(self.ckp_dir, exist_ok=True)
        self.example_dir = os.path.join(self.ckp_dir, "output")
        os.makedirs(self.example_dir, exist_ok=True)
        self.log_interval = config.log_interval
        self.save_interval = config.save_interval
        self.use_wandb = config.use_wandb
        
        self.build_model()

    def build_model(self):
        self.G = Generator(nc=self.nc, ngf=self.ngf, nz=self.nz).to(self.device)
        self.D = Discriminator(nc=self.nc, ndf=self.ndf).to(self.device)
        self.G.apply(weights_init)
        self.D.apply(weights_init)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=self.g_lr,  betas=[self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=self.d_lr,  betas=[self.beta1, self.beta2])

    def save_checkpoint(self, step):
        G_path = os.path.join(self.ckp_dir, '{}-G.pth'.format(step+1))
        D_path = os.path.join(self.ckp_dir, '{}-D.pth'.format(step+1))
        torch.save(self.G.state_dict(), G_path)
        torch.save(self.D.state_dict(), D_path)
        print('Saved model checkpoints into {}...'.format(self.ckp_dir))

    def load_checkpoint(self, resume_iters):
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.ckp_dir, '{}-G.pth'.format(resume_iters))
        D_path = os.path.join(self.ckp_dir, '{}-D.pth'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path))
        self.D.load_state_dict(torch.load(D_path))

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
    
    def train(self):
        criterion = nn.BCELoss()
        torch.manual_seed(66666)
        fixed_noise = torch.randn(32, self.nz, 1, 1, device=self.device)
        real_label = 1.
        fake_label = 0.

        iteration = 0
        if self.resume_iters:
            print("resuming step %d ..."% self.resume_iters)
            iteration = self.resume_iters
            self.load_checkpoint(self.resume_iters)


        for ep in range(self.n_epochs):
            self.G.train()  
            self.D.train() 

            D_loss_t = 0.0
            D_x_t = 0.0
            D_G_z1_t = 0.0
            G_loss_t = 0.0
            D_G_z2_t = 0.0



            for batch_idx, (real_data, _) in enumerate(self.trainset_loader):
                ################
                #   update D   #
                ################
                self.D.zero_grad()

                real_data = real_data.to(self.device)
                b_size = real_data.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)
                output = self.D(real_data).view(-1)
                D_loss_real = criterion(output, label)
                D_loss_real.backward()
                D_x = output.mean().item()
                D_x_t += D_x

                noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
                fake = self.G(noise) 
                label.fill_(fake_label)
                output = self.D(fake.detach()).view(-1)
                D_loss_fake = criterion(output, label)
                D_loss_fake.backward()
                D_G_z1 = output.mean().item()
                D_G_z1_t += D_G_z1
                D_loss = D_loss_real + D_loss_fake
                D_loss_t += D_loss.item()
                self.d_optimizer.step()

                ################
                #   update G   #
                ################
                self.G.zero_grad()
                label.fill_(real_label)
                output = self.D(fake).view(-1)
                G_loss = criterion(output, label)
                G_loss.backward()
                G_loss_t += G_loss.item()
                D_G_z2 = output.mean().item()
                D_G_z2_t += D_G_z2
                self.g_optimizer.step()

                # Output training stats
                if (iteration+1) % self.log_interval == 0:
                    print('Epoch: {:3d} [{:5d}/{:5d} ({:3.0f}%)]\tIteration: {:5d}\tD_loss: {:.6f}\tG_loss: {:.6f}\tD(x): {:.6f}\tD(G(z)): {:.6f} / {:.6f}'.format(
                        ep, (batch_idx + 1) * len(real_data), len(self.trainset_loader.dataset),
                        100. * (batch_idx + 1) / len(self.trainset_loader), iteration + 1, D_loss.item(), G_loss.item(), D_x, D_G_z1, D_G_z2))

                # Save model checkpoints
                if (iteration+1) % self.save_interval == 0 and iteration > 0:
                    self.save_checkpoint(iteration)
                    g_example = self.G(fixed_noise)
                    g_example_path = os.path.join(self.example_dir, '%d.png' % (iteration+1))
                    torchvision.utils.save_image(g_example.data, g_example_path, nrow=8, normalize=True)

                iteration += 1

            print('Epoch: {:3d} [{:5d}/{:5d} ({:3.0f}%)]\tIteration: {:5d}\tD_loss: {:.6f}\tG_loss: {:.6f}\tD(x): {:.6f}\tD(G(z)): {:.6f} / {:.6f}\n'.format(
                ep, len(self.trainset_loader.dataset), len(self.trainset_loader.dataset), 100., iteration, 
                D_loss_t / len(self.trainset_loader), G_loss_t / len(self.trainset_loader), D_x_t / len(self.trainset_loader), 
                D_G_z1_t / len(self.trainset_loader), D_G_z2_t / len(self.trainset_loader)))


            if self.use_wandb:
                import wandb
                wandb.log({"Loss_D": D_loss_t / len(self.trainset_loader),
                           "Loss_G": G_loss_t / len(self.trainset_loader),
                           "D(x)": D_x_t / len(self.trainset_loader),
                           "D(G(z1))": D_G_z1_t / len(self.trainset_loader),
                           "D(G(z2))": D_G_z2_t / len(self.trainset_loader)})
            
        self.save_checkpoint(iteration)
        g_example = self.G(fixed_noise)
        g_example_path = os.path.join(self.example_dir, '%d.png' % (iteration+1))
        torchvision.utils.save_image(g_example.data, g_example_path, nrow=8, normalize=True)

