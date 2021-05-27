## A tutorial on Autoencoders and Variational Autoencoders

On this page I will present the basic principles of autoencoders (AE) and variational autoencoders (VAE) and show how these two types of model archtectures and be implementent in Pytorch an trained on the famous MNIST dataset.

### Autoencoders (AE)

### Autencoders in PyTorch

We now learn how to setup an autoencoder in PyTorch and train in on the MNIST dataset.
The first thing we will do is to import all libraries and packages we need. We will also include GPU support in case your running this code on a machine with a GPU.

```python
#import libraries
import torch
import torch.nn as nn
import torch.Functional as F

#CPU or GPU?
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

Now we set up the encoder part of the autoencoder.

```python
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(748,512)
        self.linear2 = nn.Linear(512, latent_dim)

    def forward(self,x):
        x = torch.flatten(x, start_dim = 1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x
```

The next thing is to implement the decoder part.

```python
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dim,512)
        self.linear2 = nn.Linear(512, 784)

    def forward(self,z):
        z = self.linear1(z)
        z = F.relu(x)
        z = self.linear2(x)
        z = torch.sigmoid(z)
        z = z.reshape((-1,1,28,28))
        return z
```

Now that we have both the encoder and the decoder, we can combine the two sub-models into the final autoencoder.

```python
class AE(nn.Module):
    def __init__(self, latent_dim):
        super(AE, self).__init__()
        self.encoder = self.Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
    
    def forward(self,x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
```

We just finished the model of our first autoencoder in PyTorch! The next thing we want to do is to import the MNIST dataset and train the autoencoder. We will only implement a very quick and easy training loop and use the whole dataset for training (without splitting it into validation/test).

```python
def train(ae, data, epochs=20):
    optimizer = torch.optim.Adam(ae.parameters())
    for epoch in range(epochs):
        for x, y in data:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat = ae(x)
            loss = ((x - x_hat)**2).sum()
            loss.backward()
            optimizer.step()
    return ae
```
Once our model is trained, we can use our trained autoencoder for a couple of different tasks:

1. Input an image x to the encoder, map x to the latent space z with the help of the learned non-linear function, and decode the image to $\hat{a}$. 
2. As we are dealing with 2D data, we can visualize the latent space.
3. Sample from the latent space.

```python
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
```

![Image Autoencoder 1](https://github.com/MichaelLempart/AE-VAE/blob/gh-pages/resources/Img1_AE.JPG)

def plot_latent(ae, data, num_batches=100):
    for i, (x,  y) in enumerate(data):
        z = ae.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break
    plt.show()
![Image Autoencoder 2](https://github.com/MichaelLempart/AE-VAE/blob/gh-pages/resources/Img2_AE.JPG)

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Variational autoencoders (VAE)

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/MichaelLempart/AE-VAE/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
