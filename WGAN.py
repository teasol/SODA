from tqdm import tqdm
from scipy import io
from scipy import sparse
from datetime import datetime

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

DEVICE = "cuda:0"
BATCH_SIZE = 512
EPOCHS = 10
N_CRITIC = 5
LAMBDA_TERM = 10

def write_to_file(data):
    with open("/data/home/kimds/Output/output.txt", 'w') as f:
        f.write(data)

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
        self.device = device
        self.number_of_genes = number_of_genes
        self.model = nn.Sequential(
            # 1
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Dropout(),
            
            # 2
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(),

            # 3
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(),

            # 4
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(),

            # 5
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(),

            # 6
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(),

            # 7
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(),
            
            # 8
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Dropout(),
            
            # 9
            nn.Linear(4096, 8192),
            nn.ReLU(),
            nn.Dropout(),

            # 10
            nn.Linear(8192, self.number_of_genes)
        )
    
    def forward(self, x):
        z = self.model(x)
        return z

class Critic(nn.Module):
    def __init__(self, number_of_genes, device):
        super(Critic, self).__init__()
        self.device = device
        self.number_of_genes = number_of_genes
        self.model = nn.Sequential(
            nn.Linear(self.number_of_genes, 8192),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(),
            
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        z = self.model(x)
        return z

class WGAN_GP(object):
    def __init__(self, number_of_genes):
        self.device = DEVICE
        self.learning_rate = 1e-6
        self.n_critic = N_CRITIC
        self.n_generator = EPOCHS
        self.batch_size = BATCH_SIZE

        self.lambda_term = LAMBDA_TERM

        self.generator = Generator(number_of_genes, self.device).to(self.device)
        self.critic = Critic(number_of_genes, self.device).to(self.device)

        self.d_optimizer = optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        self.g_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)

    def train(self, train_loader):
        for g_iter in range(self.n_generator):
            if g_iter % 100 == 0:
                print(f'Generator iteration: {g_iter}/{self.n_generator}', end=' ')

            for X in train_loader:
                X = X.to(self.device)
                z = torch.randn((X.size()[0], 10)).to(self.device)

                for p in self.critic.parameters():
                    p.requires_grad = True

                for t in range(self.n_critic):
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
                
                for p in self.critic.parameters():
                    p.requires_grad = False
                
                self.generator.zero_grad()
                fake_cell = self.generator(z)
                g_loss = -self.critic(fake_cell)
                g_loss = g_loss.mean()
                g_loss.backward()
                self.g_optimizer.step()

            if g_iter % 1 == 0:
                write_to_file(f'Real Loss {d_loss_real.mean()}, Fake Loss {d_loss_fake.mean()}, W distance {W_distance}, G Loss {g_loss}')

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
        torch.save(self.generator.state_dict(), './vae_2.pkl')
        torch.save(self.critic.state_dict(), './critic_2.pkl')
        print('Models save to ./vae_2.pkl & ./critic_2.pkl ')

    def load_model(self, D_model_filename, G_model_filename):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))
        print('Generator model loaded from {}.'.format(G_model_path))
        print('Discriminator model loaded from {}-'.format(D_model_path))


if __name__ == '__main__':
    START_TIME = datetime.now()
    write_to_file('START TIME:%s'%START_TIME)
    DIR_PATH = "/data/home/kimds/Data/Normalized/"
    census = io.mmread(DIR_PATH+'census.mtx')
    write_to_file('census file loaded')
    heart = io.mmread(DIR_PATH+'heart.mtx')
    write_to_file('heart file loaded')
    immune = io.mmread(DIR_PATH+'immune.mtx')
    write_to_file('immune file loaded')
    # covid = io.mmread(DIR_PATH+'covid.mtx')


    data = sparse.hstack([census, heart, immune])
    number_of_genes = data.shape[0]

    dataset = CustomDataset(data)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    aae = WGAN_GP(number_of_genes)
    write_to_file('Train Started')
    aae.train(train_loader)