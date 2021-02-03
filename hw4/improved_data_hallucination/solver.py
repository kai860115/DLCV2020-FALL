import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.optim.lr_scheduler import StepLR
from model import Convnet, MLP, Hallucinator, Discriminator
from utils import pairwise_distances


class Solver(object):
    def __init__(self, config, train_loader, val_loader):
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.episodes_per_epoch = config.episodes_per_epoch
        self.N_way_train = config.N_way_train
        self.N_shot_train = config.N_shot_train
        self.N_query_train = config.N_query_train
        self.M_aug_train = config.M_aug_train
        self.N_way_val = config.N_way_val
        self.N_shot_val = config.N_shot_val
        self.N_query_val = config.N_query_val
        self.M_aug_val = config.M_aug_val
        self.matching_fn = config.matching_fn
        self.nz = config.nz

        self.num_epochs = config.num_epochs
        self.resume_iter = config.resume_iter
        self.num_d_steps = config.num_d_steps
        self.lr = config.lr
        self.num_steps_decay = config.num_steps_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.weight_decay = config.weight_decay
        self.active_adversarial_loss_step = config.active_adversarial_loss_step
        self.alpha_weight = config.alpha_weight
        self.exp_name = config.name
        os.makedirs(config.ckp_dir, exist_ok=True)
        self.ckp_dir = os.path.join(config.ckp_dir, self.exp_name)
        os.makedirs(self.ckp_dir, exist_ok=True)
        self.log_interval = config.log_interval
        self.ckp_interval = config.ckp_interval

        self.use_wandb = config.use_wandb
        
        self.build_model()

    def build_model(self):
        self.cnn = Convnet().to(self.device)
        self.g = Hallucinator(self.nz).to(self.device)
        self.mlp = MLP().to(self.device)
        self.d = Discriminator().to(self.device)

        self.optimizer = torch.optim.AdamW(list(self.cnn.parameters()) + list(self.g.parameters()) + list(self.mlp.parameters()), lr=self.lr,  betas=[self.beta1, self.beta2], weight_decay=self.weight_decay)
        self.optimizer_d = torch.optim.AdamW(self.d.parameters(), lr=self.lr,  betas=[self.beta1, self.beta2], weight_decay=self.weight_decay)

        if self.matching_fn == 'parametric':
            self.parametric = nn.Sequential(
                nn.Linear(800, 400),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(400, 1)
            ).to(self.device)
            self.optimizer = torch.optim.AdamW(list(self.cnn.parameters()) + list(self.g.parameters()) + list(self.mlp.parameters()) + list(self.parametric.parameters()), lr=self.lr,  betas=[self.beta1, self.beta2], weight_decay=self.weight_decay)

        self.scheduler = StepLR(self.optimizer, step_size=self.num_steps_decay, gamma=0.9)
        self.scheduler_d = StepLR(self.optimizer_d, step_size=self.num_steps_decay, gamma=0.9)

    def save_checkpoint(self, step):
        state = {'cnn': self.cnn.state_dict(),
                 'g': self.g.state_dict(),
                 'mlp': self.mlp.state_dict(),
                 'optimizer' : self.optimizer.state_dict(),
                 'd': self.d.state_dict(),
                 'optimizer_d' : self.optimizer_d.state_dict()}

        if self.matching_fn == 'parametric':
            state['parametric'] = self.parametric.state_dict()

        new_checkpoint_path = os.path.join(self.ckp_dir, '{}-improved_dhm.pth'.format(step + 1))
        torch.save(state, new_checkpoint_path)
        print('model saved to %s' % new_checkpoint_path)

    def load_checkpoint(self, resume_iter):
        print('Loading the trained models from step {}...'.format(resume_iter))
        new_checkpoint_path = os.path.join(self.ckp_dir, '{}-improved_dhm.pth'.format(resume_iter))
        state = torch.load(new_checkpoint_path)
        self.cnn.load_state_dict(state['cnn'])
        self.g.load_state_dict(state['g'])
        self.mlp.load_state_dict(state['mlp'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.d.load_state_dict(state['d'])
        self.optimizer_d.load_state_dict(state['optimizer_d'])
        if self.matching_fn == 'parametric':
            self.parametric.load_state_dict(state['parametric'])
        print('model loaded from %s' % new_checkpoint_path)
    
    def train(self):
        task_criterion = nn.CrossEntropyLoss()
        adversarial_criterion = nn.BCELoss()

        best_mean = 0
        iteration = 0
        real_label = 1.
        fake_label = 0.

        self.sample_idx_val = []
        self.noise_val = []
        for i in range(self.episodes_per_epoch):
            self.sample_idx_val.append(torch.tensor([torch.randint(self.N_shot_val * i, self.N_shot_val * (i + 1), (self.M_aug_val,)).numpy() for i in range(self.N_way_val)]).reshape(-1))
            self.noise_val.append(torch.randn((self.N_way_val * self.M_aug_val, self.nz), device=self.device))

        if self.resume_iter:
            print("resuming step %d ..."% self.resume_iter)
            iteration = self.resume_iter
            self.load_checkpoint(self.resume_iter)
            loss, mean, std = self.eval()
            if mean > best_mean:
                best_mean = mean

        episodic_acc = []

        for ep in range(self.num_epochs):
            self.cnn.train()
            self.g.train()
            self.mlp.train()
            self.d.train()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data = data.to(self.device)
                
                support_input = data[:self.N_way_train * self.N_shot_train,:,:,:] 
                query_input   = data[self.N_way_train * self.N_shot_train:,:,:,:]

                label_encoder = {target[i * self.N_shot_train] : i for i in range(self.N_way_train)}
                query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[self.N_way_train * self.N_shot_train:]])

                real_label = torch.full((self.N_way_train * self.N_shot_val,), 1., dtype=torch.float, device=self.device)
                real_label_g = torch.full((self.N_way_train * self.M_aug_train,), 1., dtype=torch.float, device=self.device)
                fake_label_g = torch.full((self.N_way_train * self.M_aug_train,), 0., dtype=torch.float, device=self.device)

                ################
                #   update D   #
                ################
                support = self.cnn(support_input)
                queries = self.cnn(query_input)

                sample_idx = torch.tensor([torch.randint(self.N_shot_train * i, self.N_shot_train * (i + 1), (self.M_aug_train,)).numpy() for i in range(self.N_way_train)]).reshape(-1)
                noise = torch.randn((self.N_way_train * self.M_aug_train, self.nz), device=self.device)

                sample = support[sample_idx]
                support_g = self.g(sample, noise)

                if ep >= self.active_adversarial_loss_step:
                    for _ in range(self.num_d_steps):
                        self.optimizer_d.zero_grad()
                        self.optimizer.zero_grad()

                        d_loss_adv_fake = adversarial_criterion(self.d(support_g.detach()).view(-1), fake_label_g)
                        d_loss_adv_real = adversarial_criterion(self.d(support.detach()).view(-1), real_label)

                        d_loss = self.alpha_weight * (d_loss_adv_fake + d_loss_adv_real)
                        d_loss.backward()
                        self.optimizer_d.step()

                else:
                    d_loss_adv_fake = torch.tensor(0).cuda()
                    d_loss_adv_real = torch.tensor(0).cuda()
                    d_loss = torch.tensor(0).cuda()
                    d_loss_task = torch.tensor(0).cuda()

                ################
                #   update H   #
                ################
                self.optimizer_d.zero_grad()
                self.optimizer.zero_grad()

                if ep >= self.active_adversarial_loss_step:
                    h_loss_adv = adversarial_criterion(self.d(support_g).view(-1), real_label_g)
                else:
                    h_loss_adv = torch.tensor(0).cuda()

                support_g_r = support_g.reshape(self.N_way_train, self.M_aug_train, -1)
                support_r = support.reshape(self.N_way_train, self.N_shot_train, -1)

                support_aug = torch.cat([support_r, support_g_r], dim=1)
                support_aug = support_aug.reshape(self.N_way_train * (self.N_shot_train + self.M_aug_train), -1)

                prototypes = self.mlp(support_aug)
                prototypes = prototypes.reshape(self.N_way_train, self.N_shot_train + self.M_aug_train, -1).mean(dim=1)
                queries = self.mlp(queries)

                if self.matching_fn == 'parametric':
                    distances = pairwise_distances(queries, prototypes, self.matching_fn, self.parametric)

                else:
                    distances = pairwise_distances(queries, prototypes, self.matching_fn)

                h_loss_task = task_criterion(-distances, query_label)
                h_loss = self.alpha_weight * h_loss_adv + h_loss_task
                h_loss.backward()
                self.optimizer.step()

                y_pred = (-distances).softmax(dim=1).max(1, keepdim=True)[1]
                episodic_acc.append(1. * y_pred.eq(query_label.view_as(y_pred)).sum().item() / len(query_label))

                if (iteration + 1) % self.log_interval == 0:
                    episodic_acc = np.array(episodic_acc)
                    mean = episodic_acc.mean()
                    std = episodic_acc.std()
                    episodic_acc = []

                    print('Epoch: {:3d} [{:d}/{:d}]  Iteration: {:5d}  h_loss: {:.4f}  h_loss_adv: {:.4f}  h_loss_task: {:.4f}  d_loss: {:.4f}  d_loss_adv_fake: {:.4f}  d_loss_adv_real: {:.4f}  Accuracy: {:.2f} +- {:.2f} %'.format(
                        ep, (batch_idx + 1), len(self.train_loader), iteration + 1, 
                        h_loss.item(), h_loss_adv.item(), h_loss_task.item(), 
                        d_loss.item(), d_loss_adv_fake.item(), d_loss_adv_real.item(),
                        mean * 100, 1.96 * std / (self.log_interval)**(1/2) * 100))

                    if self.use_wandb:
                        import wandb
                        wandb.log({
                                'h_loss': h_loss.item(),
                                'h_loss_adv': h_loss_adv.item(),
                                'h_loss_task': h_loss_task.item(),
                                'd_loss': d_loss.item(),
                                'd_loss_adv_fake': d_loss_adv_fake.item(),
                                'd_loss_adv_real': d_loss_adv_real.item(),
                                "acc_mean": mean * 100,
                                "acc_ci": 1.96 * std / (self.log_interval)**(1/2) * 100,
                                'lr': self.optimizer.param_groups[0]['lr']
                            }, step=iteration+1)



                if (iteration + 1) % self.ckp_interval == 0:
                    loss, mean, std = self.eval()
                    if mean > best_mean:
                        best_mean = mean
                        self.save_checkpoint(iteration)
                        if self.use_wandb:
                            wandb.run.summary["best_accuracy"] = best_mean * 100

                    if self.use_wandb:
                        import wandb
                        wandb.log({"val_loss": loss,
                                   "val_acc_mean": mean * 100,
                                   "val_acc_ci": 1.96 * std / (600)**(1/2) * 100}, 
                                   step=iteration+1, commit=False)

                iteration += 1

            self.scheduler.step()
            self.scheduler_d.step()

    def eval(self):
        criterion = nn.CrossEntropyLoss()
        self.cnn.eval()
        self.g.eval()
        self.mlp.eval()
        self.d.eval()
        episodic_acc = []
        loss = []
        
        with torch.no_grad():
            for b_idx, (data, target) in enumerate(self.val_loader):
                data = data.to(self.device)
                support_input = data[:self.N_way_val * self.N_shot_val,:,:,:] 
                query_input   = data[self.N_way_val * self.N_shot_val:,:,:,:]

                label_encoder = {target[i * self.N_shot_val] : i for i in range(self.N_way_val)}
                query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[self.N_way_val * self.N_shot_val:]])

                support = self.cnn(support_input)
                queries = self.cnn(query_input)

                sample_idx = self.sample_idx_val[b_idx]
                sample = support[sample_idx]
                
                noise = self.noise_val[b_idx]

                support_g = self.g(sample, noise).reshape(self.N_way_val, self.M_aug_val, -1)
                support = support.reshape(self.N_way_val, self.N_shot_val, -1)

                support_aug = torch.cat([support, support_g], dim=1)
                support_aug = support_aug.reshape(self.N_way_val * (self.N_shot_val + self.M_aug_val), -1)

                prototypes = self.mlp(support_aug)
                prototypes = prototypes.reshape(self.N_way_val, self.N_shot_val + self.M_aug_val, -1).mean(dim=1)
                queries = self.mlp(queries)

                if self.matching_fn == 'parametric':
                    distances = pairwise_distances(queries, prototypes, self.matching_fn, self.parametric)
                else:
                    distances = pairwise_distances(queries, prototypes, self.matching_fn)
                
                loss.append(criterion(-distances, query_label).item())
                y_pred = (-distances).softmax(dim=1).max(1, keepdim=True)[1]
                episodic_acc.append(1. * y_pred.eq(query_label.view_as(y_pred)).sum().item() / len(query_label))

        loss = np.array(loss)
        episodic_acc = np.array(episodic_acc)
        loss = loss.mean()
        mean = episodic_acc.mean()
        std = episodic_acc.std()

        print('\nLoss: {:.6f}  Accuracy: {:.2f} +- {:.2f} %\n'.format(loss,mean * 100, 1.96 * std / (600)**(1/2) * 100))

        return loss, mean, std
