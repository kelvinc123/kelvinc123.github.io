---
layout: post
title: "Exploring the Landscape of Generative Models: A Simple Guide"
date: 2023-11-20 16:50:16
description: Simple comparison between Autoregressive, Flow, VAEs, GANs, Energy-Based, Score-Based, and Diffusion models.
tags: VAE Autoregressive Flow Energy-Based Score-Based GAN Diffusion
categories: sample-posts
toc:
  sidebar: left
---

$$\DeclareMathOperator*{\argmin}{argmin}$$
$$\DeclareMathOperator*{\argmax}{argmax}$$

This blog offers a concise comparison of the most widely used generative models: Autoregressive Models, Normalizing Flow Models, Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), Energy-Based Models (EBMs), Score-Based Models, and the recently emerging Diffusion Models. Each of these models shared a common goal: to effectively model and represent the probability distribution $$p_{\theta}(\mathbf{x})$$ of complex, high-dimensional dataset.

## Autoregressive

Autoregressive (AR) models use the chain rule to model the probability of high-dimensional dataset. For a given high-dimensional dataset $$\mathbf{x} \in \mathbb{R}^{n}$$, the probability can be broken down into the product of conditional probabilities of **each dimension given the preceeding elements**. More formally,

$$p_{\theta}(\mathbf{x}) = \prod_{i=1}^{n}p_{\theta}(x_{i} | x_{1}, x_{2}, \dots, x_{i-1})$$

This probabilistic breakdown facilitates the application of various assumptions, like the Markov assumption, or the integration of neural network architectures such as LSTMs or Transformer models. This decomposition allows the model to easily learn the probability distribition with high accuracy. Autoregressive models have proven to be effective in generating sequential data such as in language modeling and time-series forecasting.

#### Pros

<ul>
    <li><b>Likelihood Estimation</b>: Provides the exact likelihood calculation as a result of chain rule</li>
</ul>

#### Cons

<ul>
    <li><b>Sampling speed</b>: Slow sampling speed due to the nature of sequential generation, each step requires the output from previous steps</li>
    <li><b>Scalability</b>: Struggles in handling a very large dataset and long sequences</li>
</ul>

#### Papers

<ul>
  <li><strong>MADE: Masked Autoencoder for Distribution Estimation</strong> by Mathieu Germain, Karol Gregor, Iain Murray, and Hugo Larochelle. <a href="https://arxiv.org/abs/1502.03509">Link</a>.</li>
  <li><strong>Attention Is All You Need</strong> by Ashish Vaswani et al. <a href="https://arxiv.org/abs/1706.03762">Link</a>.</li>
  <li><strong>Pixel Recurrent Neural Networks</strong> by Aaron van den Oord and others. <a href="https://arxiv.org/abs/1601.06759">Link</a>.</li>
  <li><strong>Conditional Image Generation with PixelCNN Decoders</strong> by Aaron van den Oord and others. <a href="https://arxiv.org/abs/1606.05328">Link</a>.</li>
</ul>

## Normalizing Flow Models

Normalizing Flow Models efficiently use a simple latent variable $$\mathbf{z}$$, typically Gaussian, alongside a deterministic, invertible function $$f_{\theta}$$ to model the probability distribution of the dataset $$p_{\theta}(\mathbf{x})$$. The function $$f_{\theta}$$ is defined so that $$f_{\theta}(\mathbf{z}) = \mathbf{x}$$ and $$f_{\theta}^{-1}(\mathbf{x}) = \mathbf{z}$$. The latent variable $$\mathbf{z}$$ must match the dimensionality of $$\mathbf{x}$$ to ensure that $$f_{\theta}$$ is invertible. Note that the function $$f$$ can be a composition of multiple invertible functions, $$f = f_{1} \circ f_{2} \dots \circ f_{n}$$, enabling the construction of a complex function from a simpler ones. The density of $$\mathbf{x}$$ can then be formulated using the change of variable method:

$$
p_{X}(\mathbf{x} ; \theta) = p_{Z}(\mathbf{z}) | \text{det} J_{f_{\theta}^{-1}}(\mathbf{z})|
$$

<br>
Here, $$\mathbf{z} = f_{\theta}^{-1}(\mathbf{x})$$ and $$|\text{det} J_{f^{-1}}(\mathbf{z})|$$ is the absolute value of the determinant of the jacobian matrix $$f^{-1}$$.

To evaluate the density of $$\mathbf{x}$$, the mapping function $$f$$ is used to transform $$\mathbf{x}$$ to $$\mathbf{z}$$, and use the right hand side equation above to calculate the density $$p_{X}(\mathbf{x}; \theta)$$. The simplicity of $$p_{Z}(\mathbf{z})$$ is the key to computing $$p_{X}(\mathbf{x}; \theta)$$. To generate new sample from $$p_{X}(\mathbf{x}; \theta)$$, we can sample $$\mathbf{z}$$ from the latent distribution $$p_{Z}(\mathbf{z})$$ and apply $$f$$ to obtain $$f(\mathbf{z}) = \mathbf{x}$$.

#### Pros

<ul>
  <li><b>Likelihood estimation</b>: Provides the exact likelihood calculation due to the change of variable formula</li>
</ul>

#### Cons

<ul>
  <li><b>Limitation in function selection</b>: The requirement for the mapping function f to be invertible and to maintain the same input output dimension restrict the choice of functions</li>
</ul>

#### Papers

<ul>
  <li><strong>Masked Autoregressive Flow for Density Estimation</strong> by George Papamakarios, Theo Pavlakou, and Iain Murray. <a href="https://arxiv.org/abs/1705.07057">Link</a>.</li>
  <li><strong>Normalizing Flows for Probabilistic Modeling and Inference</strong> by George Papamakarios, Eric Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, and Balaji Lakshminarayanan. <a href="https://arxiv.org/abs/1912.02762">Link</a>.</li>
  <li><strong>NICE: Non-linear Independent Components Estimation</strong> by Laurent Dinh, David Krueger, and Yoshua Bengio. <a href="https://arxiv.org/abs/1410.8516">Link</a>.</li>
  <li><strong>Gaussianization Flows</strong> by Chenlin Meng, Yang Song, Jiaming Song, and Stefano Ermon. <a href="https://arxiv.org/abs/2003.01941">Link</a>.</li>
  <li><strong>Density Estimation Using Real NVP</strong> by Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. <a href="https://arxiv.org/abs/1605.08803">Link</a>.</li>
  <li><strong>Glow: Generative Flow with Invertible 1x1 Convolutions</strong> by Diederik P. Kingma, and Prafulla Dhariwal. <a href="https://arxiv.org/abs/1807.03039">Link</a>.</li>
</ul>

## Variational Autoencoders

Variational Autoencoders (VAEs) are a class of generative model that models the probability distribution $$p_{\theta}(\mathbf{x})$$ by learning a latent representation $$\mathbf{z}$$ of the input data $$\mathbf{x}$$. They consist of two main components: an encoder that maps input data to a latent distribution, and a decoder that reconstructs the data from this latent space. Unlike Normalizing Flow Model, this latent variable $$\mathbf{z}$$ typically has smaller dimension than the input data $$\mathbf{x}$$. VAEs learn to maximize the evidence lower bound (ELBO) of the log probability of the dataset using the following formula:

$$
\begin{align*}
    \log p_{\theta}(\mathbf{x}) &\ge \textbf{ELBO}(\mathbf{x}, \theta, \phi) \\
    &= \mathbb{E}_{q_{\phi}(\mathbf{z} | \mathbf{x})}\left[\log\frac{p_{\theta}(\mathbf{x}, \mathbf{z})}{q_{\phi}(\mathbf{z} | \mathbf{x})}\right] \\
    &= \mathbb{E}_{q_{\phi}(\mathbf{z} | \mathbf{x})}[\log p_{\theta}(\mathbf{x}| \mathbf{z})] - D_{\text{KL}}(q_{\phi}(\mathbf{z} | \mathbf{x}) || p(\mathbf{z}))
\end{align*}
$$

<br>
In this setup, the $$p(\mathbf{z})$$ is usually a simple distribution like Gaussian. The term $$q_{\phi}(\mathbf{z} | \mathbf{x})$$ and $$p_{\theta}(\mathbf{x} | \mathbf{z})$$ are usually called the encoder and decoder respectively. To sample from VAEs, we first need to sample the latent $$\mathbf{z}$$ from $$p(\mathbf{z})$$ and use the decoder component $$p_{\theta}(\mathbf{x} | \mathbf{z})$$ to transform $$z$$ to the sample.

#### Pros

<ul>
  <li><b>Latent Representation</b>: Creates a meaningful lower-dimensional representation of the input data</li>
  <li><b>Sampling speed</b>: Fast sampling speed, requiring only a forward pass through the decoder</li>
</ul>

#### Cons

<ul>
  <li><b>Approximate of true distribution</b>: Doesn't exactly compute the true distribution, it uses the approximation via ELBO</li>
</ul>

#### Papers
<ul>
  <li><strong>Auto-Encoding Variational Bayes</strong> by Diederik P Kingma and Max Welling. <a href="https://arxiv.org/abs/1312.6114">Link</a>.</li>
  <li><strong>Disentangling Disentanglement in Variational Autoencoders</strong> by Emile Mathieu, Tom Rainforth, N. Siddharth, Yee Whye Teh. <a href="https://arxiv.org/abs/1812.02833">Link</a>.</li>
  <li><strong>Understanding disentangling in beta-VAE</strong> by Christopher P. Burgess, Irina Higgins, Arka Pal, Loic Matthey, Nick Watters, Guillaume Desjardins, Alexander Lerchner. <a href="https://arxiv.org/abs/1804.03599">Link</a>.</li>
</ul>

## Generative Adversarial Networks
Generative Adversarial Networks (GANs) are a generative models that are not directly model the probability distribution $$p_{\theta}(\mathbf{x})$$ of the dataset. GANs have two main components, a discriminator $$D$$ and a generator $$G$$. The discriminator acts as a binary classifier distinguishing real from generated samples, while the generator tries to fool the discriminator by generating fake samples from a random variable $$\mathbf{z}$$ drawn from a prior distribution $$p(\mathbf{z})$$. These two components can be formulated as a minimax game with the value function:

$$
\min_{G} \max_{D}V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}(\mathbf{x})}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})}[\log (1 - D(G(\mathbf{z})))]
$$

<br>
In this adversarial setup, $$G$$ and $$D$$ are updated sequentially. The discriminator sharpens its ability to distinguish real from fake, while the generator improves at creating increasingly convincing fake samples. To generate new samples, we first sample $$\mathbf{z} \sim p(\mathbf{z})$$ and then fed it into the generator. 

#### Pros
<ul>
  <li><b>Sample quality</b>: Known for producing high quality samples as a result of the minimax game</li>
  <li><b>Sampling speed</b>: Fast sampling speed, requiring only a forward pass through the generator</li>
</ul>

#### Cons
<ul>
  <li><b>Lack of probability estimation</b>: GANs do not focus maximum likelihood estimation, hence are unable to compute probabilities or bounds like VAEs</li>
  <li><b>Mode collapse</b>: Sometimes the generator only produces the limited variety of outputs</li>
  <li><b>Training complexity</b>: GANs are difficult to train, sensitive to the hyperparameters, and often face stability issues</li>
</ul>

#### Papers
<ul>
  <li><strong>Generative Adversarial Networks</strong> by Ian J. Goodfellow and colleagues. <a href="https://arxiv.org/abs/1406.2661">Link</a>.</li>
  <li><strong>Wasserstein GAN</strong> introducing a new training approach for GANs. <a href="https://arxiv.org/abs/1701.07875">Link</a>.</li>
  <li><strong>f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization</strong> by Sebastian Nowozin, Botond Cseke, Ryota Tomioka. <a href="https://arxiv.org/abs/1606.00709">Link</a>.</li>
  <li><strong>Conditional Generative Adversarial Nets</strong> by Mehdi Mirza, Simon Osindero. <a href="https://arxiv.org/abs/1411.1784">Link</a></li>
</ul>


## Energy Based Models
Energy Based Models (EBMs) estimate the probability distribution of a dataset $$p_{\theta}(\mathbf{x})$$ directly using the energy function $$E_{\theta}(\mathbf{x})$$. Unlike the other generative models, the energy function defines an unnormalized negative log-probability, which is less restrictive. To ensure that $$p_{\theta}(\mathbf{x})$$ a valid probability function, it must be nonnegative and integrate to $$1$$. Therefore, the probability density function of EBM is expressed as:

$$
p_{\theta}(\mathbf{x}) = \frac{\exp(-E_{\theta}(\mathbf{x}))}{Z_{\theta}}
$$

Where $$Z_{\theta}$$ is the normalizing constant (or partition function) of the nominator. Altough the $$Z_{\theta}$$ here is constant, it's still a function of $$\theta$$. For training EBMs, we can use maximum likelihood learning. The gradient of the $$\log p_{\theta}(\mathbf{x})$$ w.r.t $$\theta$$ is:

$$
\begin{align*}
    \nabla_{\theta}\log p_{\theta}(\mathbf{x}) &= -\nabla_{\theta}E_{\theta}(\mathbf{x}) - \mathbb{E}_{\mathbf{x}_{\text{sample}} \sim p_{\theta}(\mathbf{x})}[\nabla_{\theta}E_{\theta}(\mathbf{x}_{\text{sample}})] \\
    &\approx -\nabla_{\theta}E_{\theta}(\mathbf{x}) - \nabla_{\theta}E_{\theta}(\mathbf{x}_{\text{sample}})
\end{align*}
$$

Here, $$\mathbf{x}_{\text{sample}}$$ is drawn from our EBM $$p_{\theta}(\mathbf{x_{\text{sample}}})$$. Unfortunately, the sampling process can be challenging as it needs the MCMC methods such as Metropolish Hastings or Langevin dynamics. The iterative Langevin MCMC sampling procedure is:

$$
\mathbf{x}^{(k+1)} \leftarrow \mathbf{x}^{(k)} + \frac{\epsilon}{2} \nabla_{\mathbf{x}}\log p_{\theta}(\mathbf{x}^{(k)}) + \epsilon \mathbf{z}^{(k)}
$$

Where $$\mathbf{z}^{k}$$ is a standard Gaussian random variable. This process converges to the EBM's distribution $$p_{\theta}(\mathbf{x})$$ as $$\epsilon \rightarrow 0$$ and $$k \rightarrow \infty$$. 

#### Pros
<ul>
  <li><b>Energy function architecture</b>: Flexibility in choosing the energy function</li>
  <li><b>Direct modeling of probability</b>: Capable of modeling the probability distribution directly, without the need for an intermediate latent space</li>
  <li><b>Robust to overfitting</b>: EBMs are generally more robust to overfitting compared to other generative models</li>
</ul>

#### Cons
<ul>
  <li><b>Sampling speed</b>: The need for MCMC sampling method to be converge makes sampling very slow</li>
  <li><b>Training speed</b>: Each training iteration requires sampling, which is very slow</li>
  <li><b>Difficulty in probability estimation</b>: While EBMs model probabilities directly, calculating the partition function is often intractable</li>
</ul>

#### Papers
<ul>
  <li><strong>How to Train Your Energy-Based Models</strong> by Yang Song, Diederik P. Kingma. <a href="https://arxiv.org/abs/2101.03288">Link</a>.</li>
  <li><strong>Flow Contrastive Estimation of Energy-Based Models</strong> by Ruiqi Gao, Erik Nijkamp, Diederik P. Kingma, Zhen Xu, Andrew M. Dai, Ying Nian Wu. <a href="https://arxiv.org/abs/1912.00589">Link</a>.</li>
</ul>

## Score Based Model

#### Pros

#### Cons

#### Papers

## Diffusion Model

#### Pros

#### Cons

#### Papers
