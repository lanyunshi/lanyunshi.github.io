import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
from VAE import VAE
import GAN
import gzip

torch.manual_seed(2024)

class MNISTDataset():
    def __init__(self, data_path, train=True, transform=None):
        X, y = self.load_data(data_path, train)
        self.X = X
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        img = self.X[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.y is None:
            return img
        else:
            return img, int(self.y[index])

    def __len__(self):
        return len(self.X)

    def load_data(self, data_path, train):
        y_train = None

        if train:
            with gzip.open(data_path + '-labels.gz', 'rb') as f:
                y_train = np.frombuffer(f.read(), np.uint8, offset=8)

        with gzip.open(data_path + '-images.gz', 'rb') as f:
            x_train = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
        return x_train, y_train



def load_mnist():
    train_set = MNISTDataset(r'./data/train', train=True,
                             transform = transforms.Compose([
                                         transforms.ToTensor(), #,transforms.Normalize([0.5], [0.5])
                                         ]))
    return train_set

def deprocess_img(x):         
    # rescale image from [-1, 1] to [0, 1]
    return (x + 1.0) / 2.0     



def loss_function(reconstruction_function, recon_x, x, mu, logvar):
    #print(recon_x[:3, :3], x[:3, :3])
    BCE = reconstruction_function(recon_x, x)
    KLD = -0.5 * torch.sum(1 + 2*logvar - torch.exp(2*logvar) - mu**2)
    return BCE + KLD

def train_with_VAE(train_set):
    #print(test_set[0]); exit()
    _, height, width = train_set[0][0].shape
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    model = VAE(width*height, 50, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
    loss_func = nn.BCELoss(reduction = 'sum')
    recons = []

    num_epochs = 40
    for epoch in range(1, num_epochs + 1):
        train_l_sum, train_acc_sum, n = 0., 0., 0
        for i, Xy in enumerate(train_dataloader):
            #if i> 100: continue
            X, y = Xy
            X = X.squeeze(1).view(-1, width * height)
            recon_batch, mu, logvar = model(X)
            loss = loss_function(loss_func, recon_batch, X, mu, logvar)

            # 梯度清零
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_l_sum += loss.item()
            n += y.shape[0]

        if epoch % 10 == 0:
            print('epoch %d, loss %.4f' % (epoch, train_l_sum / n))
            recon_batch = recon_batch.view(-1, height, width).detach().numpy()

    recon, row_recon = [], []
    for i in range(len(recon_batch)):
        recon += [recon_batch[i, :, :]]
        if (i + 1) % 8 == 0:
            row_recon += [np.concatenate(recon, 1)]
            recon = []
    row_recon = np.concatenate(row_recon, 0)
    # print('row_recon', row_recon.shape); exit()

    fig = plt.figure()
    plt.imshow(row_recon, cmap='gray', interpolation='none')
    plt.title("Generated Images")
    plt.xticks([])
    plt.yticks([])
    plt.show()


def discriminator_loss(loss_fn, logits_real, logits_fake):
    size = logits_real.shape[0]
    true_labels = torch.ones(size, 1).float()
    false_labels = torch.zeros(size, 1).float()
    loss = loss_fn(logits_real, true_labels) + loss_fn(logits_fake, false_labels)
    return loss

def generator_loss(loss_fn, logits_fake):
    size = logits_fake.shape[0]
    true_labels = torch.ones(size, 1).float()
    loss = loss_fn(logits_fake, true_labels)
    return loss

def train_with_GAN(train_set):
    #print(test_set[0]); exit()
    noise_dim = 96
    _, height, width = train_set[0][0].shape
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    discriminator = GAN.Discriminator(width*height, 256, 256)
    generator = GAN.Generator(noise_dim, 1024, width*height)
    dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=5e-4, betas=(0.5, 0.999))
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=5e-4, betas=(0.5, 0.999))
    dis_loss_func = nn.BCELoss(reduction = 'sum')
    gen_loss_func = nn.BCELoss(reduction = 'sum')
    recons = []

    num_epochs = 5
    for epoch in range(1, num_epochs + 1):
        train_d_sum, train_g_sum, n = 0., 0., 0
        for i, Xy in enumerate(train_dataloader):
            #if i> 100: continue
            X, y = Xy
            batch_size = X.size(0)
            X = X.squeeze(1).view(-1, width * height)

            # 训练判别网络
            logits_real = discriminator(X)
            rand_noise = (torch.rand(batch_size, noise_dim) - 0.5)/0.5
            fake_images = generator(rand_noise)
            logits_fake = discriminator(fake_images)

            d_error = discriminator_loss(dis_loss_func, logits_real, logits_fake)
            dis_optimizer.zero_grad()
            d_error.backward()
            dis_optimizer.step()

            # 训练生成网络
            rand_noise = (torch.rand(batch_size, noise_dim) - 0.5)/0.5
            fake_images = generator(rand_noise)

            gen_logits_fake = discriminator(fake_images)
            g_error = generator_loss(gen_loss_func, gen_logits_fake)
            gen_optimizer.zero_grad()
            g_error.backward()
            gen_optimizer.step()

            train_d_sum += d_error.item()
            train_g_sum += g_error.item()
            n += y.shape[0]

        print('epoch %d, discriminator loss %.4f, generator loss %.4f' % (epoch, train_d_sum/n, train_g_sum/n))

    recon_batch = fake_images.view(-1, height, width).detach().numpy()
    recon_batch = deprocess_img(recon_batch)

    recon, row_recon = [], []
    for i in range(len(recon_batch)):
        recon += [recon_batch[i, :, :]]
        if (i + 1) % 8 == 0:
            row_recon += [np.concatenate(recon, 1)]
            recon = []
    row_recon = np.concatenate(row_recon, 0)

    fig = plt.figure()
    plt.imshow(row_recon, cmap='gray', interpolation='none')
    plt.title("Generated Images")
    plt.xticks([])
    plt.yticks([])
    plt.show()

if __name__ == '__main__':
    train_set = load_mnist()

    train_with_VAE(train_set)
    
    train_with_GAN(train_set)

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_feature, h1, h2):
        super(VAE, self).__init__()
        '''
        The inference network has structure:
        h1 = ReLU(W1 x + b1)
        mu = W2 h1 + b2
        log sigma = W3 h1 + b3

        The generation network has structure:
        h1 = ReLU(W1 z + b1)
        x_{hat} = sigmoid(W2 h1 + b2)
        '''
        self.fc1 = nn.Linear(input_feature, h1)
        self.fc21 = nn.Linear(h1, h2)
        self.fc22 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h1)
        self.fc4 = nn.Linear(h1, input_feature)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        '''
        Sampling z via reparameterize:
        z = mu + sigma * epsilon
        '''
        std = 0.5 * torch.exp(logvar)
        z = torch.randn(std.size()) * std + mu
        return z

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class Discriminator(nn.Module):
    def __init__(self, input_feature, h1, h2):
        super(Discriminator, self).__init__()
        '''
        The discriminator has structure:
        h1 = LeakyReLU(W1 x + b1)
        h2 = LeakyReLU(W2 h1 + b2)
        h3 = sigmoid(W3 h2 + b3)
        '''
        self.fc1 = nn.Linear(input_feature, h1)
        self.act1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(h1, h2)
        self.act2 = nn.LeakyReLU(0.2)
        self.fc3 = nn.Linear(h2, 1)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x

class Generator(nn.Module):
    def __init__(self, noise_dim, h1, h2):
        super(Generator, self).__init__()
        '''
        The generator has structure:
        h1 = ReLU(W1 z + b1)
        h2 = ReLU(W2 h1 + b2)
        x_{hat} = tanh(W3 h2 + b3)
        '''
        self.fc1 = nn.Linear(noise_dim, h1)
        self.fc2 = nn.Linear(h1, h1)
        self.fc3 = nn.Linear(h1, h2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        return x



