
## A tutorial on Autoencoders and Variational Autoencoders

On this page I will present the basic principles of autoencoders (AE) and variational autoencoders (VAE) and show how these two types of model archtectures and be implementent in Pytorch an trained on the famous MNIST dataset.

### Autoencoders (AE)

Autoencoders are neural networks that can perform dimensional reduction of data. Through optimization, the network learns a non-linear transformation from in example an image space X, to a latent space Z, and back from latent space Z to space X.
This can be achieved by an encoder and decoder like model architecture. The encoder encodes X to Z, and the decoder decodes Z to X. 


<img src=
"(https://github.com/MichaelLempart/AE-VAE/blob/gh-pages/resources/Img0_AE.JP">

Autoencoders can be used for several different types of application ,like in example:
- dimensional reduction
- feature extraction: train the AE, extract latent space features and feed them into another neural network

Nevertheless, there is a problem with autoencoders: It does not learn or generate a smooth and contious latent space. We will see the exact difference when implementing autoencoders and compare them to variational autoencoders by visualizing the latent space.

Now, lets dive right into it and code our first autoencoder in PyTorch.

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
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+p%28z%7Cx%29+%3D+%5Cfrac%7Bp%28x%2Cz%7D%7Bp%28x%29%7D+%3D+%5Cfrac%7Bp%28x%2Cz%7D%7B+%5Cintop_%7Bz%7Dp%28x%2Cz%29%5Cmathrm%7Bd%7Dz%7D" 
alt="p(z|x) = \frac{p(x,z}{p(x)} = \frac{p(x,z}{ \intop_{z}p(x,z)\mathrm{d}z}">

Again, the integral in instractable. By using some mathematical properties of conditional probabilty and log properties we can rewrite the KL-divergence:

<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle++D_%7BKL%7D%28q%7C%7Cp%29+%3D++%5Cintop_%7Bz%7Dq%28z%7Cx%29log%5Cfrac%7Bq%28z%7Cx%29%7D%7Bp%28z%7Cx%29%7D%5Cmathrm%7Bd%7Dz+%3D++%5Cintop_%7Bz%7Dq%28z%7Cx%29log%28%5Cfrac%7Bq%28z%7Cx%29%7D%7Bp%28x%2Cz%29%7Dp%28x%29%29%5Cmathrm%7Bd%7Dz+%3D+%5Cintop_%7Bz%7Dq%28z%7Cx%29%5Blog%5Cfrac%7Bq%28z%7Cx%29%7D%7Bp%28x%2Cz%29%7D+%2B+log+p%28x%29%5D%5Cmathrm%7Bd%7Dz++%3D++%5Cintop_%7Bz%7Dq%28z%7Cx%29%5Blog%5Cfrac%7Bq%28z%7Cx%29%7D%7Bp%28x%2Cz%29%7D%5D%5Cmathrm%7Bd%7Dz+%2B+log+p%28x%29++%5Cintop_%7Bz%7Dq%28z%7Cx%29%5Cmathrm%7Bd%7Dz" 
alt=" D_{KL}(q||p) =  \intop_{z}q(z|x)log\frac{q(z|x)}{p(z|x)}\mathrm{d}z =  \intop_{z}q(z|x)log(\frac{q(z|x)}{p(x,z)}p(x))\mathrm{d}z = \intop_{z}q(z|x)[log\frac{q(z|x)}{p(x,z)} + log p(x)]\mathrm{d}z  =  \intop_{z}q(z|x)[log\frac{q(z|x)}{p(x,z)}]\mathrm{d}z + log p(x)  \intop_{z}q(z|x)\mathrm{d}z">

Wow.... What a mess... Nevertheless, the positive thing is that we know can get rid of the last term <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+%5Cintop_%7Bz%7Dq%28z%7Cx%29%5Cmathrm%7Bd%7Dz" 
alt="\intop_{z}q(z|x)\mathrm{d}z">.

As we know that a normalized probability distribution always integrates to 1, this term becomes 1 as well and we are left with:

<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle++D_%7BKL%7D%28q%7C%7Cp%29+%3D+++%5Cintop_%7Bz%7Dq%28z%7Cx%29%5Blog%5Cfrac%7Bq%28z%7Cx%29%7D%7Bp%28x%2Cz%29%7D%5D%5Cmathrm%7Bd%7Dz+%2B+%5Clog%7Bp%28x%29%7D" 
alt=" D_{KL}(q||p) =   \intop_{z}q(z|x)[log\frac{q(z|x)}{p(x,z)}]\mathrm{d}z + \log{p(x)}">

We can now call the first term for L, which is is also known as the varitational free energy or the upper bound to <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+-+%5Clog%7Bp%28x%29%7D" 
alt="- \log{p(x)}"> :

<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+L+%3D+++%5Cintop_%7Bz%7Dq%28z%7Cx%29%5Blog%5Cfrac%7Bq%28z%7Cx%29%7D%7Bp%28x%2Cz%29%7D%5D%5Cmathrm%7Bd%7Dz" 
alt="L =   \intop_{z}q(z|x)[log\frac{q(z|x)}{p(x,z)}]\mathrm{d}z">

The KL-divergence might then be written as:

<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+D_%7BKL%7D%28q%7C%7Cp%29+%3D++L+%2B+%5Clog%7Bp%28x%29%7D" 
alt="D_{KL}(q||p) =  L + \log{p(x)}">

Finally, we define -L as the Evidence Lower Bound (ELBO): 

<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+-+L+%3D+%5Clog%7Bp%28x%29%7D+-+D_%7BKL%7D%28q%7C%7Cp%29+%3D+%5Carg+%5Cmin%7BL%7D+%3D+%5Carg+%5Cmax%7B-L%7D" 
alt="- L = \log{p(x)} - D_{KL}(q||p) = \arg \min{L} = \arg \max{-L}">

From the previous mentioned properties of the KL-divergence, we know that the KL-divergence cannot be negative, it is zero or greater than this. That meansm that we can bound the KL-divergence at 0 and we can minimize the KL-divergence by mimizing the variational free energy or by maximizing the ELBO:

<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%5Carg+%5Cmin%7BL%7DD_%7BKL%7D%28q%7C%7Cp%29+%3D+%5Carg+%5Cmin%7BL%7D+%3D+%5Carg+%5Cmax%7B-L%7D" 
alt="\arg \min{L}D_{KL}(q||p) = \arg \min{L} = \arg \max{-L}">

Mimizing the KL-divergence means that we are able to find the parameters that make our original probability distribution and our variational probability distribution similar.

Now we can re-write the variational free energy L somewhat more, by replacing the definition of conditional probability and replacing it by the joint probability:

<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+L+%3D++%5Cintop_%7Bz%7Dq%28z%7Cx%29%5B%5Clog%7B%5Cfrac%7Bq%28z%7Cx%29%7D%7Bp%28x%2Cz%29%7D%7D%5D%5Cmathrm%7Bd%7Dz+%3D+%5Cintop_%7Bz%7Dq%28z%7Cx%29%5B%5Clog%7B%5Cfrac%7Bq%28z%7Cx%29%7D%7Bp%28x%7Cz%29p%28z%29%7D%7D%5D%5Cmathrm%7Bd%7Dz" 
alt="L =  \intop_{z}q(z|x)[\log{\frac{q(z|x)}{p(x,z)}}]\mathrm{d}z = \intop_{z}q(z|x)[\log{\frac{q(z|x)}{p(x|z)p(z)}}]\mathrm{d}z">

If we know use the properties of the logarithm, we can split the above term into two integrals:

<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+L+%3D++%5Cintop_%7Bz%7Dq%28z%7Cx%29%5Clog%7B%5Cfrac%7Bq%28z%7Cx%29%7D%7Bp%28z%29%7D%7D%5Cmathrm%7Bd%7Dz+-+%5Cintop_%7Bz%7Dq%28z%7Cx%29%5Clog%7Bp%28x%7Cz%29%7D%5Cmathrm%7Bd%7Dz" 
alt="L =  \intop_{z}q(z|x)\log{\frac{q(z|x)}{p(z)}}\mathrm{d}z - \intop_{z}q(z|x)\log{p(x|z)}\mathrm{d}z">

And, believe it or not, this was the final step in the derivation of the VAE loss function:

<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+L+%3D+D_%7BKL%7D%28q%28z%7Cx%29%7C%7Cp%28z%29%29-%5Cmathbb%7BE%7D_%7Bz%5Cbacksim+q%28z%7Cx%29%7D%5Clog%7Bp%28x%7Cz%29%7D" 
alt="L = D_{KL}(q(z|x)||p(z))-\mathbb{E}_{z\backsim q(z|x)}\log{p(x|z)}">

By analyzing the two terms, we can see that the first one is again a KL-divergence, where <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+q%28z%7Cx%29" 
alt="q(z|x)"> is the variational density (the encoder) and <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+q%28z%29" 
alt="q(z)"> a prior density over the latent variables. So by minimizing the KL-divergence, we make the overall density in the latent space equal to our prior distribution, which we design or choose on our own.

The second part of the above equation is the expectation value over the latent space (the output of the encoder).

Ok, lets recap.... We input an image X into our encoder, which maps X to a density <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+q%28z%7Cx%29" 
alt="q(z|x)">. By showing different samples X, we try to make <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+q%28z%7Cx%29" 
alt="q(z|x)"> as similar as possible to our prior distribution <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+p%28z%29" 
alt="p(z)"> wich we design. The decoder <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+p%28x%7Cz%29" 
alt="p(x|z)"> going from latent space to image space is used to evaluate the probability of seeing sample X given the latent variabels <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+z%28p%28x%7Cz%29%29" 
alt="z(p(x|z))">, which is the log-likelihood.

So how do we choose our variational distribution? We could in example use a Gaussian model:

<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+q%28z%7Cx%29+%3D+%5Cmathcal%7BN%7D%28z%3B+%5Cmu%28x%29%2C+diag%28%5Csigma%5E2%29%29+%3D+%5Cfrac%7B1%7D%7B+%5Csqrt%7B2%5Cpi%7D+%5Cprod_%7Bd%7D%5E%7Bi%3D1%7D+%5Csigma_i%7D%5Cexp%5B-%5Cfrac%7B1%7D%7B2%7D+%5Csum%5Cnolimits_%7Bi%3D1%7D%5E%7Bd%7D%28%5Cfrac%7B+x_i+-+%5Cmu_i%7D%7B%5Csigma_i%7D%29%5E2%5D+" 
alt="q(z|x) = \mathcal{N}(z; \mu(x), diag(\sigma^2)) = \frac{1}{ \sqrt{2\pi} \prod_{d}^{i=1} \sigma_i}\exp[-\frac{1}{2} \sum\nolimits_{i=1}^{d}(\frac{ x_i - \mu_i}{\sigma_i})^2] ">

and we choose our prior as a standard normal distribution:

<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+p%28z%29+%3D+%5Cmathcal%7BN%7D%280%2C1%29" 
alt="p(z) = \mathcal{N}(0,1)">

The latent variable z has a mean vector and a diagonal variance matrix, which are the outputs of the encoder. In addtion, we assume that our prior is a standard normal distribution. 

For the case of a Gaussian, the KL-divergence is given as:

<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+D_%7BKL%7D%28q%28z%7Cx%29%7C%7Cp%28z%29%29+%3D++%5Cintop_%7Bz%7Dq%28z%7Cx%29%5Clog%7B%5Cfrac%7Bq%28z%7Cx%29%7D%7Bp%28z%29%7D%7D%5Cmathrm%7Bd%7Dz+%3D+%5Cfrac%7B1%7D%7B2%7D++%5Csum%5Cnolimits_%7Bi%3D1%7D%5E%7Bd%7D%281%2B%5Clog%7B%5Csigma_i%5E2%28x%29+-+%5Cmu_i%5E2%28x%29+-+%5Csigma_i%5E2%28x%29%29%7D+" 
alt="D_{KL}(q(z|x)||p(z)) =  \intop_{z}q(z|x)\log{\frac{q(z|x)}{p(z)}}\mathrm{d}z = \frac{1}{2}  \sum\nolimits_{i=1}^{d}(1+\log{\sigma_i^2(x) - \mu_i^2(x) - \sigma_i^2(x))} ">

Lets write our derived loss function, using the parameters <img src=
"https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5Ctheta%0A" 
alt="\theta
"> we want to optimize:

<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+L+%3D++-+%5Cmathbb%7BE%7D_%7Bz%5Cbacksim+q%28z%7Cx%29%7D%5Clog%7Bp%28x%7Cz%2C%5Ctheta_%7Bdecoder%7D%29%7D+%2B+D_%7BKL%7D%28q%28z%7Cx%2C%5Ctheta_%7Bencoder%7D%29%7C%7Cp%28z%29%29+" 
alt="L =  - \mathbb{E}_{z\backsim q(z|x)}\log{p(x|z,\theta_{decoder})} + D_{KL}(q(z|x,\theta_{encoder})||p(z)) ">

One final problem arises. We said that we sample z from the probability distribution <img src=
"https://render.githubusercontent.com/render/math?math=%5Ctextstyle+q%28z%7Cx%29%0A" 
alt="q(z|x)
">. Unfortunately, the process of sampling from a probability distribution is not differntiable. We can't compute the derivative of a sampling distribution with respect to it's input. Nevertheless, we can use the so called parametrization trick. Instead of sampling from <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+q%28z%7Cx%29" 
alt="q(z|x)">, we sample from a unit Gaussian distribution, which has zero mean and a standard deviation of 1. This is still a random sampling process, but the unit Gaussian does not have any parameters that we want to optimize. When we ccompute the latent variable z, we can scale and shift it by the mean and the variance of the unit Gaussian.

- take a unit Gaussian <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%5Cepsilon%5Cbacksim%5Cmathcal%7BN%7D%280%2C1%29%0A" 
alt="\epsilon\backsim\mathcal{N}(0,1)
">
- shift the unit Gaussian by the mean value and and scale it. Mean and variance come from the latent distribution <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+z+%3D+%5Cmu%28x%29+%2B+%5Csigma%28x%29+%5Codot+%5Cepsilon+%0A" 
alt="z = \mu(x) + \sigma(x) \odot \epsilon 
">

This can be visualized like this:

![Image](https://github.com/MichaelLempart/AE-VAE/blob/gh-pages/resources/reparametrization_trick.jpg)


The gray fields are deterministic, while the blue ones are random. Without using the parametrization trick, a random field (or node), is blocking the backpropagation flow. If we now replace the random node by our unit Gaussian and the shift and scale operations, we can see that we get a non-blocked, continues backpropagation path. We still can't compute the gradient of the unit Gaussian, but we don't care! There are no parameters in the unit Gaussian which we want to change or optimize. 

The last term we haven't talked about is the expectation value of making the observation x given the latent vector z and the decoder parameters <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+%5Ctheta" 
alt="\theta">:

<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%5Cmathbb%7BE%7D_%7Bz%5Cbacksim+q%28z%7Cx%29%7D%5Clog%7Bp%28x%7Cz%3B%5Ctheta_%7Bdecoder%7D%29%7D+%3D+%5Clog%7Bp%28x%7Cz%3B%5Ctheta_%7Bdecoder%7D%29%7D" 
alt="\mathbb{E}_{z\backsim q(z|x)}\log{p(x|z;\theta_{decoder})} = \log{p(x|z;\theta_{decoder})}">

As we are using the stoachistic gradient descend to optimize our model, we replace the expected value by a single sample.
Now we can see that we have a Maximum likelihood problem, which can be used to find the parameters of a probability distribution. Depending on the problem we are working on, we have to decide on the type of distribution. In the MNIST case, where pixels are binary (0 or 1), we us a sigmoid output layer. This can be thought of as a Bernoulli distribution, where the negative log-likelihood of the Bernoulli distribution is the binary cross-entropy (our reconstruction loss):

<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+L%28%5Ctheta%29+%3D++%5Csum%5Cnolimits_%7Bi%3D1%7D%5E%7BN%7Dx_i%5Clog%7B%5Chat%7Bx_i%7D%28x%2C%5Ctheta%29%7D%2B%281-x_i%29%5Clog%7B%5B1-%5Chat%7Bx_i%7D%28x%2C%5Ctheta%29%5D%7D" 
alt="L(\theta) =  \sum\nolimits_{i=1}^{N}x_i\log{\hat{x_i}(x,\theta)}+(1-x_i)\log{[1-\hat{x_i}(x,\theta)]}">

With that, the final loss function for the MNIST dataset becomes:

<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+L+%3D+D_%7BKL%7D%28q%28z%7Cx%29%7C%7Cp%28z%29%29-%5Cmathbb%7BE%7D_%7Bz%5Cbacksim+q%28z%7Cx%29%7D%5Clog%7Bp%28x%7Cz%29%7D+%3D+%5Cfrac%7B1%7D%7B2%7D++%5Csum%5Cnolimits_%7Bi%3D1%7D%5E%7Bd%7D%281%2B%5Clog%7B%5Csigma_i%5E2%28x%29+-+%5Cmu_i%5E2%28x%29+-+%5Csigma_i%5E2%28x%29%29%7D+-+%5Csum%5Cnolimits_%7Bi%3D1%7D%5E%7BN%7Dx_i%5Clog%7B%5Chat%7Bx_i%7D%28x%2C%5Ctheta%29%7D%2B%281-x_i%29%5Clog%7B%5B1-%5Chat%7Bx_i%7D%28x%2C%5Ctheta%29%5D%7D" 
alt="L = D_{KL}(q(z|x)||p(z))-\mathbb{E}_{z\backsim q(z|x)}\log{p(x|z)} = \frac{1}{2}  \sum\nolimits_{i=1}^{d}(1+\log{\sigma_i^2(x) - \mu_i^2(x) - \sigma_i^2(x))} - \sum\nolimits_{i=1}^{N}x_i\log{\hat{x_i}(x,\theta)}+(1-x_i)\log{[1-\hat{x_i}(x,\theta)]}">

That was more than a little bit of math and I hope you stayed with me during this section. If not, all you need is the above loss function and some knowledge about the functional principle of a variational autoencoder in order to implement it. Now, lets get our hands dirty and implement a VAE in PyTorch :-)
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
