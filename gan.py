

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import matplotlib.pyplot as plt

training_data = datasets.MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transforms.ToTensor())

from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data,batch_size=32,shuffle=True)

class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
        nn.Conv2d(1, 64, 4, 2, 1, bias=False),       # 28x28 -> 14x14
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(64, 128, 4, 2, 1, bias=False),     # 14x14 -> 7x7
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(128, 1, 7, 1, 0, bias=False),      # 7x7 -> 1x1 (valid)
        nn.Sigmoid()
    )

  def forward(self, x):
    return self.net(x).view(-1,1)

class Generator(nn.Module):
  def __init__(self, z_dim=100):
    super().__init__()
    self.net = nn.Sequential(
        nn.ConvTranspose2d(z_dim, 128, 7, 1, 0, bias=False),  # 1x1 -> 7x7
        nn.BatchNorm2d(128),
        nn.ReLU(True),

        nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),     # 7x7 -> 14x14
        nn.BatchNorm2d(64),
        nn.ReLU(True),

        nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),       # 14x14 -> 28x28
        nn.Tanh()
    )

  def forward(self, x):
    return self.net(x)

gen = Generator()
disc = Discriminator()

criterion = nn.BCELoss()
lr = 2e-3
opt_G = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

import torchvision.utils as vutils

fixed_noise = torch.randn(64,100,1,1)

fake = gen(fixed_noise).detach()
grid = vutils.make_grid(fake, padding=2, normalize=True)
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title(f"Generated Images at Epoch 0")
plt.imshow(grid.permute(1, 2, 0))
plt.show()


for epoch in range(50):
  for i, (real_images,_) in enumerate(train_dataloader):
      real_image = real_images
      b_size = real_image.size(0)

      real_labels = torch.ones(b_size,1)
      fake_labels = torch.zeros(b_size,1)

      fake_img = torch.randn(b_size,100,1,1)
      fake_images = gen(fake_img).detach()

      disc_real_loss = criterion(disc(real_image),real_labels)
      disc_fake_loss = criterion(disc(fake_images),fake_labels)
      disc_loss = (disc_real_loss + disc_fake_loss)

      opt_D.zero_grad()
      disc_loss.backward()
      opt_D.step()

      # Generator
      z = torch.randn(b_size,100,1,1)
      fakes = gen(z)
      gen_loss = criterion(disc(fakes),real_labels)

      opt_G.zero_grad()
      gen_loss.backward()
      opt_G.step()


  print(f"Epoch {epoch+1}/{50} | D Loss: {disc_loss.item():.4f} | G Loss: {gen_loss.item():.4f}")

  with torch.no_grad():
        fake = gen(fixed_noise).detach()
        grid = vutils.make_grid(fake, padding=2, normalize=True)
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title(f"Generated Images at Epoch {epoch+1}")
        plt.imshow(grid.permute(1, 2, 0))
        plt.show()

