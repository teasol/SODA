from tqdm import tqdm
from scipy import io
from scipy import sparse
import scipy
import gzip
import scanpy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.distributions.beta import Beta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle

class CustomDataset(Dataset):
    def __init__(self, mtx):
        super(CustomDataset, self).__init__()
        self.mtx = mtx.tocsc()
        self.number_of_genes, self.number_of_cells = mtx.shape
    def __len__(self):
        return self.number_of_cells

    def __getitem__(self, idx):
        X = torch.FloatTensor(np.asarray(self.mtx[:, idx].todense()).squeeze())        
        return X

class Generator(nn.Module):
    def __init__(self, number_of_genes, device):
        super(Generator, self).__init__()
        self.number_of_genes = number_of_genes
        self.model = nn.Sequential(
            nn.Linear(10, 512),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(2048, self.number_of_genes)
        )
    
    def forward(self, x):
        z = self.model(x)
        return z

class Critic(nn.Module):
    def __init__(self, number_of_genes, device):
        super(Critic, self).__init__()
        self.number_of_genes = number_of_genes
        self.model = nn.Sequential(
            nn.Linear(self.number_of_genes, 2048),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        z = self.model(x)
        return z

class WGAN_GP(object):
    def __init__(self, number_of_genes):
        self.device = "cuda:0"
        self.learning_rate = 1e-6
        self.n_critic = 5
        self.n_generator = 10000
        self.batch_size = 32

        self.lambda_term = 100

        self.generator = Generator(number_of_genes, self.device).to(self.device)
        self.critic = Critic(number_of_genes, self.device).to(self.device)

        self.d_optimizer = optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        self.g_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)

    def train(self, train_loader):
        for g_iter in range(self.n_generator):
            if g_iter % 100 == 0:
                print(f'Generator iteration: {g_iter}/{self.n_generator}', end=' ')

            for p in self.critic.parameters():
                p.requires_grad = True

            for t in range(self.n_critic):
                X = next(iter(train_loader))
                X = X.to(self.device)
                z = torch.randn((X.size()[0], 10)).to(self.device)
                self.critic.zero_grad()

                d_loss_real = self.critic(X)

                fake_cell = self.generator(z)
                d_loss_fake = self.critic(fake_cell)

                gradient_penalty = self.calculate_gradient_penalty(X, fake_cell)

                d_loss = d_loss_fake - d_loss_real + gradient_penalty
                W_distance = (torch.abs(d_loss_real - d_loss_fake)).sum()
                d_loss = d_loss.mean()
                d_loss.backward()
                self.d_optimizer.step()
                
                if g_iter % 100 == 0:
                    print(f'Real Loss {d_loss_real.mean()}, Fake Loss {d_loss_fake.mean()}, W distance {W_distance}', end=' ')
            for p in self.critic.parameters():
                p.requires_grad = False

            X = next(iter(train_loader))
            X = X.to(self.device)
            z = torch.randn((X.size()[0], 10)).to(self.device)
            
            self.generator.zero_grad()
            fake_cell = self.generator(z)
            g_loss = -self.critic(fake_cell)
            g_loss = g_loss.mean()
            g_loss.backward()
            self.g_optimizer.step()

            if g_iter % 100 == 0:
                print(f'G Loss {g_loss}')

        self.save_model()

    def calculate_gradient_penalty(self, real_X, fake_X):
        eta = torch.FloatTensor(1).uniform_(0,1)
        eta = eta.expand(real_X.size())
        eta = eta.to(self.device)

        interpolated = eta * real_X + ((1 - eta) * fake_X)
        interpolated = interpolated.to(self.device)
        interpolated = Variable(interpolated, requires_grad=True)

        prob_interpolated = self.critic(interpolated)

        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0]
        
        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2) * self.lambda_term
        
        return grad_penalty

    def save_model(self):
        torch.save(self.generator.state_dict(), './vae.pkl')
        torch.save(self.critic.state_dict(), './critic.pkl')
        print('Models save to ./vae.pkl & ./critic.pkl ')

    def load_model(self, D_model_filename, G_model_filename):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))
        print('Generator model loaded from {}.'.format(G_model_path))
        print('Discriminator model loaded from {}-'.format(D_model_path))


DIR_PATH = "/data/home/kimds/Data/Normalized/"
census = io.mmread(DIR_PATH+'census.mtx')
heart = io.mmread(DIR_PATH+'heart.mtx')
immune = io.mmread(DIR_PATH+'immune.mtx')
# covid = io.mmread(DIR_PATH+'covid.mtx')


data = sparse.hstack([census, heart, immune])
number_of_genes = data.shape[0]

dataset = CustomDataset(data)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)


aae = WGAN_GP(number_of_genes)
aae.train(train_loader)