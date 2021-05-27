#import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt

#CPU or GPU?
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Encoder_VAE(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder_VAE, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dim)
        self.linear3 = nn.Linear(512, latent_dim)

        self.N = torch.distributions.Normal(0,1)
        #self.N.loc = self.N.loc.cuda() #for GPU
        #self.N.scale = self.N.scale.cuda()

        self.kl = 0
    
    def forward(self, x):
        x = torch.flatten(x, start_dim = 1)
        x = self.linear1(x)
        x = F.relu(x)
        #get the mean values (one for each latent space dimension)
        mu = self.linear2(x)
        #get sigma
        sigma = self.linear3(x)
        sigma = torch.exp(sigma)
        #Reparametrization trick (sample from standard notmal distribution and scale and shift)
        z = mu + sigma*self.N.sample(mu.shape)

        #compute the KL divergence
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()

        return z
    
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(784,512)
        self.linear2 = nn.Linear(512, latent_dim)

    def forward(self,x):
        x = torch.flatten(x, start_dim = 1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dim,512)
        self.linear2 = nn.Linear(512, 784)

    def forward(self,z):
        z = self.linear1(z)
        z = F.relu(z)
        z = self.linear2(z)
        z = torch.sigmoid(z)
        z = z.reshape((-1,1,28,28))
        return z

class AE(nn.Module):
    def __init__(self, latent_dim):
        super(AE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
    
    def forward(self,x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder_VAE(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

def train(ae, data, epochs=20):
    optimizer = torch.optim.Adam(ae.parameters())
    for epoch in range(epochs):
        loss_per_epoch = []
        print('training epoch:', epoch)
        for x, y in data:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat = ae(x)
            loss = ((x - x_hat)**2).sum()
            loss_per_epoch.append(loss.to('cpu').detach().numpy())
            loss.backward()
            optimizer.step()
        print('epoch {} done. Mean loss : {}'.format(epoch, np.mean(loss_per_epoch)))
    return ae

def train_vae(vae, data, epochs=20):
    optimizer = torch.optim.Adam(vae.parameters())
    for epoch in range(epochs):
        loss_per_epoch = []
        print('training epoch:', epoch)
        for x, y in data:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat = vae(x)
            #Add the KL divergence to the cost function
            loss = ((x - x_hat)**2).sum() + vae.encoder.kl
            loss_per_epoch.append(loss.to('cpu').detach().numpy())
            loss.backward()
            optimizer.step()
        print('epoch {} done. Mean loss : {}'.format(epoch, np.mean(loss_per_epoch)))
    return vae

def plot_latent(ae, data, num_batches=100):
    for i, (x,  y) in enumerate(data):
        z = ae.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break
    plt.show()

def reconstruct_image(ae, r0=(-5, 10), r1=(-10, 5), n=12):
    w = 28
    img = np.zeros((n*w, n*w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = ae.decoder(z)
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
    plt.imshow(img, extent=[*r0, *r1])
    plt.show()

def interpolate(autoencoder, x_1, x_2, n=12):
    z_1 = autoencoder.encoder(x_1)
    z_2 = autoencoder.encoder(x_2)
    z = torch.stack([z_1 + (z_2 - z_1)*t for t in np.linspace(0, 1, n)])
    interpolate_list = autoencoder.decoder(z)
    interpolate_list = interpolate_list.to('cpu').detach().numpy()

    w = 28
    img = np.zeros((w, n*w))
    for i, x_hat in enumerate(interpolate_list):
        img[:, i*w:(i+1)*w] = x_hat.reshape(28, 28)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

from PIL import Image

def interpolate_gif(autoencoder, filename, x_1, x_2, n=100):
    z_1 = autoencoder.encoder(x_1)
    z_2 = autoencoder.encoder(x_2)

    z = torch.stack([z_1 + (z_2 - z_1)*t for t in np.linspace(0, 1, n)])

    interpolate_list = autoencoder.decoder(z)
    interpolate_list = interpolate_list.to('cpu').detach().numpy()*255

    images_list = [Image.fromarray(img.reshape(28, 28)).resize((256, 256)) for img in interpolate_list]
    images_list = images_list + images_list[::-1] # loop back beginning

    images_list[0].save(
        f'{filename}.gif',
        save_all=True,
        append_images=images_list[1:],
        loop=1)



def main():

    latent_dim = 2
    #autoencoder = AE(latent_dim).to(device) # GPU
    vae = VAE(latent_dim).to((device))

    data = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./data', transform=torchvision.transforms.ToTensor(),download=True),
            batch_size=128,
            shuffle=True)

    #autoencoder = train(autoencoder, data)
    vae = train_vae(vae, data)

    #plot_latent(autoencoder, data)
    plot_latent(vae, data)

    #reconstruct_image(autoencoder)
    reconstruct_image(vae, r0=(-3, 3), r1=(-3, 3))

    x, y = data.__iter__().next() # hack to grab a batch
    x_1 = x[y == 1][1].to(device) # find a 1
    x_2 = x[y == 0][1].to(device) # find a 0

    interpolate(vae, x_1, x_2, n=20)

    #interpolate(autoencoder, x_1, x_2, n=20)

    interpolate_gif(vae, "vae", x_1, x_2)

if __name__ == '__main__':

    main()