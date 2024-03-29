

import torch
import torch.nn as nn
from torchinfo import summary

class Discriminator(nn.Module):
    def __init__(self, input_channels, discriminator_features):
        super(Discriminator, self).__init__()
        self.name="discriminator"
        self.discrimi = nn.Sequential(
            nn.Conv2d(input_channels, discriminator_features, kernel_size=3, stride=2, padding=1),
            # no bacth normalisation in the first layer of discriminator
            nn.LeakyReLU(0.2),
            self.ConvIntoBatchnormIntoLeakyReLU(discriminator_features, discriminator_features * 2, 4, 2, 1),
            self.ConvIntoBatchnormIntoLeakyReLU(discriminator_features* 2, discriminator_features * 4, 4, 2, 1),
            self.ConvIntoBatchnormIntoLeakyReLU(discriminator_features * 4, discriminator_features * 8, 4, 2, 1),
            nn.Conv2d(discriminator_features * 8, 1, kernel_size=4, stride=2, padding=0),
            #pass through sigmoid to get a value between 0 and 1
            nn.Sigmoid(),
        )

    def ConvIntoBatchnormIntoLeakyReLU(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False,)
            ,nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.discrimi(x)


class Generator(nn.Module):
    def __init__(self, noise, input_channels, generator_features):
        super(Generator, self).__init__()
        self.name="generator"
        self.net = nn.Sequential(
            self.ConvIntoBatchnormIntoReLU(noise, generator_features * 16, 4, 1, 0),
            self.ConvIntoBatchnormIntoReLU(generator_features * 16, generator_features * 8, 4, 2, 1),
            self.ConvIntoBatchnormIntoReLU(generator_features * 8, generator_features * 4, 4, 2, 1),
            self.ConvIntoBatchnormIntoReLU(generator_features * 4, generator_features * 2, 4, 2, 1),
            #no batch normalization in the last layer
            nn.ConvTranspose2d(generator_features * 2, input_channels, kernel_size=3, stride=2, padding=1),
            #to generate a value between [-1,1]
            nn.Tanh(),
        )

    def ConvIntoBatchnormIntoReLU(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride,padding,bias=False,),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 854, 3, 220, 220
    noise_channels = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 54)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    generate = Generator(noise_channels, in_channels, 54)
    z = torch.randn((N, noise_channels, 1, 1))
    assert generate(z).shape == (N, in_channels, H, W), "Generator test failed"
    print("Success, tests passed!")


if __name__ == "__main__":

    model = Discriminator(3,64)
    print(summary(model))

import os
from PIL import Image
import torch
from torchvision import transforms

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform=None):
        self.images = []
        self.transform = transform

        # Get a sorted list of filenames
        filenames = sorted(os.listdir(folder_path), key=lambda x: int(x.split('.')[0]))

        # Iterate through each image in the folder
        for filename in filenames:
            # Construct the full path to the image
            image_path = os.path.join(folder_path, filename)
            # Open image
            image = Image.open(image_path)
            # Apply transform if provided
            if self.transform:
                image = self.transform(image)
            self.images.append(image)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]

!pip install tensorflow[and-cuda]

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


#go fast brrrr
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(123)
lr = 2e-4
batch_size = 128
img_channels = 3
noise_channels = 100
num_epochs = 50
num_features_discriminator = 54
num_features_generateerator = 54



transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize images to a fixed size
    transforms.ToTensor(),           # Convert images to PyTorch tensors
        transforms.Normalize(
            [0.5 for _ in range(3)], [0.5 for _ in range(3)]
        ),
    ]
)

dataset = CustomDataset("/kaggle/input/landscape-image-colorization/landscape Images/gray", transform=transforms)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

generate = Generator(noise_channels, img_channels, num_features_generateerator).to(device)

discriminate = Discriminator(img_channels, num_features_discriminator).to(device)

initialize_weights(generate)
initialize_weights(discriminate)

opt_generate = optim.Adam(generate.parameters(), lr=lr, betas=(0.5, 0.999))
opt_disc = optim.Adam(discriminate.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(32, noise_channels, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

generate.train()
discriminate.train()

for epoch in range(num_epochs):
    # Target labels not needed! <3 unsupervised
    for batch_idx, real in enumerate(dataloader):
        real = real.to(device)
        noise = torch.randn(batch_size, noise_channels, 1, 1).to(device)
        fake = generate(noise)

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        disc_real = discriminate(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = discriminate(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        output = discriminate(fake).reshape(-1)
        loss_generate = criterion(output, torch.ones_like(output))
        generate.zero_grad()
        loss_generate.backward()
        opt_generate.step()

        # Print losses occasionally and print to tensorboard
        if epoch % 10 == 0 and batch_idx%100==0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(dataloader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_generate:.4f}"
            )

            with torch.no_grad():
                fake = generate(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:3], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:3], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)

                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                show(img_grid_real,img_grid_fake)

            step += 1

import matplotlib.pyplot as plt
import numpy as np



def show(img_real,img_fake):
    npimgr = img_real.cpu().numpy()



    npimgf = img_fake.cpu().numpy()



    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(np.transpose(npimgr, (1,2,0)), interpolation='nearest')
    f.add_subplot(1,2, 2)
    plt.imshow(np.transpose(npimgf, (1,2,0)), interpolation='nearest')
    plt.show(block=True)
