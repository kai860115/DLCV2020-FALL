import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from model import VAE


class Solver(object):
    def __init__(self, trainset_loader, config):
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.trainset_loader = trainset_loader
        self.nz = config.nz
        self.n_epochs = config.n_epochs
        self.resume_iters = config.resume_iters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.kld_factor = config.kld_factor
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
        self.model = VAE(z_dim=self.nz).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,  betas=[self.beta1, self.beta2] )

    def save_checkpoint(self, step):
        state = {'state_dict': self.model.state_dict(),
                 'optimizer' : self.optimizer.state_dict()}
        new_checkpoint_path = os.path.join(self.ckp_dir, '{}-vae.pth'.format(step + 1))
        torch.save(state, new_checkpoint_path)
        print('model saved to %s' % new_checkpoint_path)

    def load_checkpoint(self, resume_iters):
        print('Loading the trained models from step {}...'.format(resume_iters))
        new_checkpoint_path = os.path.join(self.ckp_dir, '{}-vae.pth'.format(resume_iters))
        state = torch.load(new_checkpoint_path)
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])
        print('model loaded from %s' % new_checkpoint_path)
    
    def train(self):
        iteration = 0
        torch.manual_seed(66666)
        fixed_noise = torch.randn((32, self.nz)).cuda()

        if self.resume_iters:
            print("resuming step %d ..."% self.resume_iters)
            iteration = self.resume_iters
            self.load_checkpoint(self.resume_iters)

        for ep in range(self.n_epochs):
            self.model.train()  # set training mode

            mse_loss_t = 0.0
            kld_loss_t = 0.0

            for batch_idx, (data, _) in enumerate(self.trainset_loader):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                rec, mu, logvar = self.model(data)
                mse_loss = F.mse_loss(rec, data)
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                mse_loss_t += mse_loss.item()
                kld_loss_t += kld_loss.item()

                loss = mse_loss + self.kld_factor * kld_loss
                loss.backward()

                self.optimizer.step()

                if (iteration + 1) % self.log_interval == 0:
                    print('Epoch: {:3d} [{:5d}/{:5d} ({:3.0f}%)]\tIteration: {:5d}\tMSE: {:.6f}\tKLD: {:.6f}'.format(
                        ep, (batch_idx + 1) * len(data), len(self.trainset_loader.dataset),
                        100. * (batch_idx + 1) / len(self.trainset_loader), iteration + 1, mse_loss.item(), kld_loss.item()))

                if (iteration + 1) % self.save_interval == 0 and iteration > 0:
                    self.save_checkpoint(iteration)
                    g_example = self.model.decode(fixed_noise)
                    g_example_path = os.path.join(self.example_dir, '%d.png' % (iteration+1))
                    torchvision.utils.save_image(g_example.data, g_example_path, nrow=8, normalize=True)

                iteration += 1

            print('Epoch: {:3d} [{:5d}/{:5d} ({:3.0f}%)]\tIteration: {:5d}\tMSE: {:.6f}\tKLD: {:.6f}\n'.format(
                ep, len(self.trainset_loader.dataset), len(self.trainset_loader.dataset), 100., iteration,
                mse_loss_t / len(self.trainset_loader), kld_loss_t / len(self.trainset_loader)))

            if self.use_wandb:
                import wandb
                wandb.log({"MSE": mse_loss_t / len(self.trainset_loader),
                           "KLD": kld_loss_t / len(self.trainset_loader)})
            

        self.save_checkpoint(iteration)
        g_example = self.model.decode(fixed_noise)
        g_example_path = os.path.join(self.example_dir, '%d.png' % (iteration+1))
        torchvision.utils.save_image(g_example.data, g_example_path, nrow=8, normalize=True)

