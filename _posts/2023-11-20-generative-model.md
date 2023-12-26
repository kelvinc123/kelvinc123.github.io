---
layout: post
title:  "Exploring the Landscape of Generative Models: A Comparative Guide"
date:   2023-11-20 16:50:16
description: Comprehensive comparison between Autoregressive, Flow, VAEs, GANs, Energy-Based, Score-Based, and Diffusion models.
tags: VAE Autoregressive Flow Energy-Based Score-Based GAN Diffusion
categories: sample-posts
toc:
  sidebar: left
---

$$\DeclareMathOperator*{\argmin}{argmin}$$
$$\DeclareMathOperator*{\argmax}{argmax}$$

This blog offers a concise comparison of the most widely used generative models: Autoregressive Models, Normalizing Flow Models, Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), Energy-Based Models (EBMs), Score-Based Models, and the recently emerging Diffusion Models. Each of these models shared a common goal: to effectively model and represent the probability distribution $$p_{\theta}(\mathbf{x})$$ of complex, high-dimensional dataset. 

## Autoregressive
Autoregressive (AR) model uses the chain rule to model the probability of high-dimensional dataset. For any given high-dimensional dataset $$\mathbf{x} \in \mathbb{R}^{n}$$, the probability can be broken down into the product of conditional probabilities of **each dimension given the preceeding elements**. More formally,

$$p_{\theta}(\mathbf{x}) = \prod_{i=1}^{n}p_{\theta}(x_{i} | x_{1}, x_{2}, \dots, x_{i-1})$$

This probabilistic breakdown facilitates the application of various assumptions, like the Markov assumption, or the integration of neural network architectures such as LSTMs or Transformer models. This decomposition allows the model to easily learn the probability distribition with high accuracy. Autoregressive models have proven to be effective in generating sequential data such as in language modeling and time-series forecasting. 

### Pros
<ul>
    <li><b>Likelihood Estimation</b>: Provides the exact likelihood calculation as a result of chain rule</li>
</ul>

### Cons
<ul>
    <li><b>Sampling speed</b>: Slow sampling speed due to the nature of sequential generation, each step requires the output from previous steps</li>
    <li><b>Scalability</b>: Struggles in handling a very large dataset and long sequences</li>
</ul>

### Papers
<ul>
  <li><strong>MADE: Masked Autoencoder for Distribution Estimation</strong> by Mathieu Germain, Karol Gregor, Iain Murray, and Hugo Larochelle. <a href="https://arxiv.org/abs/1502.03509">Link</a>.</li>
  <li><strong>Attention Is All You Need</strong> by Ashish Vaswani et al. <a href="https://arxiv.org/abs/1706.03762">Link</a>.</li>
  <li><strong>Pixel Recurrent Neural Networks</strong> by Aaron van den Oord and others. <a href="https://arxiv.org/abs/1601.06759">Link</a>.</li>
  <li><strong>Conditional Image Generation with PixelCNN Decoders</strong> by Aaron van den Oord and others. <a href="https://arxiv.org/abs/1606.05328">Link</a>.</li>
</ul>


## Normalizing Flow Models
Normalizing Flow Model use a straightforward latent distribution, typically gaussian, denoted as $$\mathbf{z}$$ to model the probability distribution of the dataset $$p_{\theta}(\mathbf{x})$$. They employ a deterministic, invertible function $$f_{\theta}$$ such that $$f_{\theta}(\mathbf{z}) = \mathbf{x}$$ and $$f_{\theta}^{-1}(\mathbf{x}) = \mathbf{z}$$. The latent variable $$\mathbf{z}$$ must match the dimensionality of $$\mathbf{x}$$ to ensure that $$f$$ is invertible. The function $$f$$ can be a composition of multiple invertible functions, $$f = f_{1} \circ f_{2} \dots \circ f_{n}$$, enabling the construction of a complex function from a simpler ones. The density of $$\mathbf{x}$$ can be formulated using the change of variable method:

$$
p_{X}(\mathbf{x} ; \theta) = p_{Z}(\mathbf{z}) | \text{det} J_{f_{\theta}^{-1}}(\mathbf{z})|
$$

<br>
Here, $$\mathbf{z} = f_{\theta}^{-1}(\mathbf{x})$$ and $$|\text{det} J_{f^{-1}}(\mathbf{z})|$$ is the absolute value of the determinant of the jacobian matrix $$f^{-1}$$. 

To evaluate the density of $$\mathbf{x}$$, the mapping function $$f$$ is used to transform $$\mathbf{x}$$ to $$\mathbf{z}$$, and use the right hand side equation above to calculate the density $$p_{X}(\mathbf{x}; \theta)$$. The simplicity of $$p_{Z}(\mathbf{z})$$ is the key to computing $$p_{X}(\mathbf{x}; \theta)$$. To generate new sample from $$p_{X}(\mathbf{x}; \theta)$$, we can sample $$\mathbf{z}$$ from prior distribution $$p_{Z}(\mathbf{z})$$ and apply $$f$$ to obtain $$f(\mathbf{z}) = \mathbf{x}$$. 

### Pros
<ul>
  <li><b>Likelihood Estimation</b>: Provides the exact likelihood calculation due to the change of variable formula</li>
</ul>

### Cons
<ul>
  <li><b>Limitation in function selection</b>: The requirement for the mapping function f to be invertible and to maintain the same input output dimension restrict the choice of functions</li>
</ul>

### Papers
<ul>
  <li><strong>Masked Autoregressive Flow for Density Estimation</strong> by George Papamakarios, Theo Pavlakou, and Iain Murray. <a href="https://arxiv.org/abs/1705.07057">Link</a>.</li>
  <li><strong>Normalizing Flows for Probabilistic Modeling and Inference</strong> by George Papamakarios, Eric Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, and Balaji Lakshminarayanan. <a href="https://arxiv.org/abs/1912.02762">Link</a>.</li>
  <li><strong>NICE: Non-linear Independent Components Estimation</strong> by Laurent Dinh, David Krueger, and Yoshua Bengio. <a href="https://arxiv.org/abs/1410.8516">Link</a>.</li>
  <li><strong>Gaussianization Flows</strong> by Chenlin Meng, Yang Song, Jiaming Song, and Stefano Ermon. <a href="https://arxiv.org/abs/2003.01941">Link</a>.</li>
  <li><strong>Density Estimation Using Real NVP</strong> by Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. <a href="https://arxiv.org/abs/1605.08803">Link</a>.</li>
</ul>

## Variational Autoencoder

### Pros

### Cons

### Papers

## Generative Adversarial Networks

### Pros

### Cons

### Papers

## Energy Based Model

### Pros

### Cons

### Papers

## Score Based Model

### Pros

### Cons

### Papers

## Diffusion Model

### Pros

### Cons

### Papers