## A tutorial on Autoencoders and Variational Autoencoders

On this page I will present the basic principles of autoencoders (AE) and variational autoencoders (VAE) and show implementations in Pytorch using the MNIST dataset.

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
class AE(nn-Module):
    def __init__(self, latent_dim):
        super(AE, self).__init__()
        self.encoder = self.Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
    
    def forward(self,x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
```
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
