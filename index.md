
## A tutorial on Autoencoders and Variational Autoencoders

On this page I will present the basic principles of autoencoders (AE) and variational autoencoders (VAE) and show how these two types of model archtectures and be implementent in Pytorch an trained on the famous MNIST dataset.

### Autoencoders (AE)

![Image](https://github.com/MichaelLempart/AE-VAE/blob/gh-pages/resources/Img0_AE.JPG)

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

1. Input an image x to the encoder, map x to the latent space z with the help of the learned non-linear function, and decode the image to <img src=
"https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5Chat%7Bx%7D" 
alt="\hat{x}">.
2. As we are dealing with a 2-dimensional latent space, we can visualize the latent space by using a scatter plot.
3. Finally, we can sample from the latent space.

Lets start by visualizing the latent space with a scatter plot.

```python
def plot_latent(ae, data, num_batches=100):
    for i, (x,  y) in enumerate(data):
        z = ae.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break
    plt.show()
```
![Image](https://github.com/MichaelLempart/AE-VAE/blob/gh-pages/resources/Img1_AE.JPG)

We see that similar digits are plottet next to each other, i.eg. the digit 0 and the digit 6.

From the 2D plot, we got a grasp on where the latent space is. Lets use this knowledge and sample from a the space.


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

![Image](https://github.com/MichaelLempart/AE-VAE/blob/gh-pages/resources/Img2_AE.JPG)

We can see that we actually got digit looking images, even though not all of them are perfect.
Another cool thing is that the digit position actually corresponds to the position in the 2D scatter plot.

### Variational Autoencoders (VAE)

Lets take a look at variational autoencoders or short VAE. A VAE is a so called generative model, belonging to the field of generative learning, where we observe a probability function by samples and try to learn this probability distribution as well as a way to create completely new samples. 

While the normal autoencoder describes the latent space in a deterministic way, a VAE describes the latent space in a probabilistic way. Instead of just mapping an input image x to a latent space vector z, a VAE maps the input image x to a probabilistic representation (i.eg. a vector of means and variances/uncertainties). For each given input, aach latent space attribute is represented by a probability distribution. When we want to decode the image, we randomly sample from the latent space distribution, which generates a vector z that can be used by the decoder of the VAE.

If we would repeat the sampling process twice for the same input image, we would end up with two images that are very similar to the input, but also very similar to each other. This is because we still try to minimize the reconstruction error between the input and the output of the VAE, leadning to a smooth latent space representation.

![Image](https://github.com/MichaelLempart/AE-VAE/blob/gh-pages/resources/Img4_VAE.jpg)

This is not the case with a normal autoencoder, where the latent space might not be smooth. Two latent space points that are close to each other, might result in two totally different output images, or two very similar input images might map into competlety different points in the latent space. In a VAE, we try to avoid this.

The image below shows the architecture of a VAE. Like the normal autoencoder, the model consists of an encoder and a decoder part. 

![Image](https://github.com/MichaelLempart/AE-VAE/blob/gh-pages/resources/Img0_VAE.JPG)

### A little bit of math....

If you want to go straight into the implementation of a VAE in PyTorch, you can skip this part. If you are interested in the math behind a VAE, fasten your belt and continue reading....

The key concept of a variation autoencoder is "variational inference", which is a mathematical tool used to approximate a probability distribution by another distributution that is easy to sample from.

The VAE consists of hidden variables z and visible variables x (i.eg. our images). We want to say something about the probability distribution of the latent space variabels z, given samples from x (this is called for "inference" of "infer characteristics z from x").
To model the probability distribution z given x, We can use Bayes rule of probability:

<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%7Bp%28z%7Cx%29%7D++%3D++%5Cfrac%7Bp%28x%7Cz%29p%28z%29%7D%7Bp%28x%29%7D" 
alt="{p(z|x)}  =  \frac{p(x|z)p(z)}{p(x)}">

The problem is that <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+p%28x%29" 
alt="p(x)"> is extremely difficult to evaluate (if even possible):

<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+p%28x%29%3D%5Cintop_%7Bz%7D+p%28x%7Cz%29p%28z%29%5Cmathrm%7Bd%7Dz+" 
alt="p(x)=\intop_{z} p(x|z)p(z)\mathrm{d}z ">

The above integral is intractable...
One way to solve this issue is by using variational inference, mentionend in the beginning. In variational inference, we replace <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+p%28z%7Cx%29" 
alt="p(z|x)"> by <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+q%28z%7Cx%29" 
alt="q(z|x)">, which is a distribution we know and that we can evaluate. This new distribution is called for "variational distribution". 
The goal is now to make the variational distribution as close to the original distribution as possible. For that, we parametrize <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+q%28z%7Cx%29" 
alt="q(z|x)"> by some learnable parameters (therefore we called it variational distribution) <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+%5Ctheta" 
alt="\theta"> --> <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+q%28z%7Cx%3B%5Ctheta%29" 
alt="q(z|x;\theta)">.

But how can we measure the similarity between two probability distributions?
The most common metrics for this is the Kullback-Leibner divergence (KL-divergence), also called for realtive entropy, which measure the distance between two distributions:

<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle++D_%7BKL%7D%28q%7C%7Cp%29+%3D++D_%7BKL%7D%28q%28z%7Cx%29%7C%7Cp%28z%7Cx%29%29+%3D++%5Cintop_%7Bx%7Dq%28x%29log%5Cfrac%7Bq%28x%29%7D%7Bp%28x%29%7D%5Cmathrm%7Bd%7Dx+" 
alt=" D_{KL}(q||p) =  D_{KL}(q(z|x)||p(z|x)) =  \intop_{x}q(x)log\frac{q(x)}{p(x)}\mathrm{d}x ">

This can be seen as the expaction with respect to the logarithmic difference between the two distributions.
The KL-divergence has some nice properties:
- KL is nonnegative, meaning it's always greater then 0 or 0, if the two distributions are exactly the same
- The KL-divergence is not symmetric

Nevertheless, we still have a problem. We still don't know <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+p%28z%7Cx%29" 
alt="p(z|x)">. Using the rule of multiplication we can write:

<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+p%28z%7Cx%29%3D%5Cfrac%7Bp%28x%2Cz%29%7D%7Bp%28x%29%7D%0A" 
alt="p(z|x)=\frac{p(x,z)}{p(x)}
">
<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+p%28x%29+%3D+++%5Cintop_%7Bz%7Dp%28x%2Cz%29%5Cmathrm%7Bd%7Dz" 
alt="p(x) =   \intop_{z}p(x,z)\mathrm{d}z">


### Variational autencoders in PyTorch

Now that we know how to build an autoencoder in PyTorch, lets code a variational autoencder.
As before, we start with the encoder part of the VAE, which is different from the autoencoder that we have seen before.

Now we have to add the mean and the standard deviation, and we have to implement the reparametrization trick, so that we can use backpropagation.
We will also add the KL divergence term in the encoder class and add the loss to the reconstruction loss.

```python
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
``` 

The decoder part of the VAE is exactly the same as before and we can just copy and past the code.

```python
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
```

Combining the variational encoder and the decoder, gives us the complete model of a variational autoencoder.

```python
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder_VAE(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
``` 

Now that we finished our model, we have to slightly change the training loop. As we have seen before, the loss function of the variational autoencder consist of two terms, the reconstruction loss and the KL divergence. We already used the reconstruction loss when we trained the normal autoencder. We can simply take the previous training loop and att the KL divergence term to the loss.

```python
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
````

After training, we repeat our experiments. We create a 2D scatterplot of the latent space, and we sample from the latent space.

![Image](https://github.com/MichaelLempart/AE-VAE/blob/gh-pages/resources/Img1_VAE.JPG)

We can observe that the latent space variabels are much closer to each other and seem to be more similar to a Gaussian distribution. Like for the autoencoder, digits that are similar are mapped next to each other. Overall the latent space is much smoother and more compact when compared to the autoencoder.

![Image](https://github.com/MichaelLempart/AE-VAE/blob/gh-pages/resources/Img2_VAE.JPG)

When we sample from the latent space, we can generate digit-like images, which position correspond to the position of the 2D scatter plot.

### Latent space interpolation

Another interesting thing is that we can interpolating linearly between two latent variables given their corresponding input images.

```python
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
```

![Image](https://github.com/MichaelLempart/AE-VAE/blob/gh-pages/resources/Img3_VAE.JPG)
