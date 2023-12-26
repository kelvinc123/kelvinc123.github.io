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

### Pros

### Cons

### Papers

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