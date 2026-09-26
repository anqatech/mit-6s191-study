# Lecture 01 — Intro to Deep Learning

## Status

- [x] Watched the lecture video
- [ ] Completed unaided recall
- [ ] Reviewed the slides
- [x] Consolidated reference notes
- [ ] Completed implementation exercises
- [ ] Completed the review quiz

## Resources

- [Course page](https://introtodeeplearning.com/)
- [Lecture slides](https://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L1.pdf)

## Initial recall

_To be completed before consulting polished notes or revisiting the slides._

## Reference notes

These notes capture the mathematical backbone of the lecture. They are intended
as a compact reference, not as a complete derivation of every result.

### 1. The basic computational unit

For an input vector $\mathbf{x}\in\mathbb{R}^{d}$, a single neuron first forms
an affine combination

$$
z = \mathbf{w}^{\mathsf T}\mathbf{x}+b,
$$

then applies an activation function $g$:

$$
\hat y = g(z) = g\!\left(\mathbf{w}^{\mathsf T}\mathbf{x}+b\right).
$$

- $\mathbf{w}$ contains the learnable weights.
- $b$ is the learnable bias.
- $z$ is the pre-activation (or logit when used for classification).
- $\hat y$ is the neuron's output.

The bias can be absorbed into the weight vector by augmenting the input with a
constant $1$, but keeping it separate is usually clearer in code.

> **Terminology.** A classical perceptron uses a hard threshold and is not
> differentiable at that threshold. The lecture uses *perceptron* more broadly
> for a neuron with a differentiable nonlinear activation. Our first notebook
> therefore implements a **sigmoid neuron**.

### 2. Activation functions

Activation functions introduce nonlinearity. Without them, composing any
number of dense layers still produces only another affine transformation, so
depth would not increase the class of functions the network can represent.

#### Sigmoid

$$
\sigma(z)=\frac{1}{1+e^{-z}},
\qquad
\sigma'(z)=\sigma(z)\bigl(1-\sigma(z)\bigr).
$$

Its output lies in $(0,1)$, which makes it natural for a binary probability.
For large $|z|$, its derivative approaches zero, which can produce vanishing
gradients.

#### Hyperbolic tangent

$$
\tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}},
\qquad
\frac{d}{dz}\tanh(z)=1-\tanh^2(z).
$$

Its output lies in $(-1,1)$ and is centred around zero, but it can also
saturate.

#### Rectified Linear Unit (ReLU)

$$
\operatorname{ReLU}(z)=\max(0,z),
\qquad
\operatorname{ReLU}'(z)=
\begin{cases}
1, & z>0,\\
0, & z<0.
\end{cases}
$$

At $z=0$ the derivative is convention-dependent; software frameworks choose a
subgradient. ReLU is simple and does not saturate for positive inputs.

### 3. Dense layers and neural networks

A dense layer computes several neurons simultaneously. With column-vector
notation,

$$
\mathbf{z}^{(\ell)}
=\mathbf{W}^{(\ell)}\mathbf{a}^{(\ell-1)}+\mathbf{b}^{(\ell)},
\qquad
\mathbf{a}^{(\ell)}=g^{(\ell)}\!\left(\mathbf{z}^{(\ell)}\right),
$$

where $\mathbf{a}^{(0)}=\mathbf{x}$. If layer $\ell-1$ has $n_{\ell-1}$
units and layer $\ell$ has $n_\ell$ units, then

$$
\mathbf{W}^{(\ell)}\in\mathbb{R}^{n_\ell\times n_{\ell-1}},
\qquad
\mathbf{b}^{(\ell)},\mathbf{z}^{(\ell)},\mathbf{a}^{(\ell)}
\in\mathbb{R}^{n_\ell}.
$$

A deep neural network is a composition of these transformations:

$$
\hat{\mathbf{y}}=f(\mathbf{x};\theta),
$$

where $\theta$ denotes every weight and bias in the network. Hidden layers can
learn intermediate features; their nonlinear activations allow the network to
form nonlinear decision boundaries.

### 4. Measuring prediction error

For one example $(\mathbf{x}^{(i)},\mathbf{y}^{(i)})$, the loss is

$$
\mathcal{L}\!\left(f(\mathbf{x}^{(i)};\theta),\mathbf{y}^{(i)}\right).
$$

The empirical loss, or objective function, is the mean over $N$ examples:

$$
J(\theta)=\frac{1}{N}\sum_{i=1}^{N}
\mathcal{L}\!\left(f(\mathbf{x}^{(i)};\theta),\mathbf{y}^{(i)}\right).
$$

#### Binary cross-entropy (binary classification)

For $y\in\{0,1\}$ and a predicted probability $\hat y\in(0,1)$,

$$
\mathcal{L}_{\mathrm{BCE}}(\hat y,y)
=-\left[y\log(\hat y)+(1-y)\log(1-\hat y)\right].
$$

It strongly penalizes confident predictions that are wrong. In practical code,
a combined *BCE-with-logits* operation is numerically more stable than applying
sigmoid and logarithms separately.

#### Mean squared error (regression)

$$
\mathcal{L}_{\mathrm{MSE}}(\hat{\mathbf y},\mathbf y)
=\frac{1}{m}\sum_{j=1}^{m}(y_j-\hat y_j)^2.
$$

MSE is appropriate when the target is a continuous quantity rather than a
class probability.

### 5. Training as optimization

Learning means choosing parameters that minimize empirical loss:

$$
\theta^*=\underset{\theta}{\operatorname{argmin}}\;J(\theta).
$$

Gradient descent repeatedly moves the parameters in the direction of steepest
decrease:

$$
\theta \leftarrow \theta-\eta\nabla_\theta J(\theta),
$$

where $\eta>0$ is the learning rate.

- If $\eta$ is too small, learning is slow and can stall.
- If $\eta$ is too large, updates may overshoot or diverge.
- Adaptive optimizers such as Adam adjust effective learning rates during
  training, often separately for each parameter.

### 6. Backpropagation and the chain rule

Backpropagation is an efficient application of the chain rule from the loss
backwards through the network. If a parameter $w$ affects a hidden value $z$,
which affects the prediction $\hat y$, then

$$
\frac{\partial J}{\partial w}
=\frac{\partial J}{\partial \hat y}
 \frac{\partial \hat y}{\partial z}
 \frac{\partial z}{\partial w}.
$$

For a multilayer network, define the local error signal
$\boldsymbol{\delta}^{(\ell)}=\partial\mathcal{L}/\partial\mathbf{z}^{(\ell)}$.
Moving backwards through a hidden layer gives

$$
\boldsymbol{\delta}^{(\ell)}
=\left(\mathbf{W}^{(\ell+1)}\right)^{\mathsf T}
 \boldsymbol{\delta}^{(\ell+1)}
 \odot g^{(\ell)\prime}\!\left(\mathbf{z}^{(\ell)}\right),
$$

and the parameter gradients for one example are

$$
\frac{\partial\mathcal{L}}{\partial\mathbf{W}^{(\ell)}}
=\boldsymbol{\delta}^{(\ell)}
 \left(\mathbf{a}^{(\ell-1)}\right)^{\mathsf T},
\qquad
\frac{\partial\mathcal{L}}{\partial\mathbf{b}^{(\ell)}}
=\boldsymbol{\delta}^{(\ell)}.
$$

#### Useful result for our first notebook

For a sigmoid output $\hat y=\sigma(z)$ paired with binary cross-entropy,

$$
\frac{\partial\mathcal{L}}{\partial z}
=\frac{\partial\mathcal{L}}{\partial\hat y}
 \frac{\partial\hat y}{\partial z}
=\hat y-y.
$$

Since $z=\mathbf{w}^{\mathsf T}\mathbf{x}+b$, it follows immediately that

$$
\frac{\partial\mathcal{L}}{\partial\mathbf{w}}
=(\hat y-y)\mathbf{x},
\qquad
\frac{\partial\mathcal{L}}{\partial b}=\hat y-y.
$$

This cancellation is one reason sigmoid and binary cross-entropy form a useful
pair for binary classification.

### 7. Full-batch, stochastic, and mini-batch gradients

The exact full-dataset gradient is

$$
\nabla_\theta J(\theta)
=\frac{1}{N}\sum_{i=1}^{N}\nabla_\theta\mathcal{L}^{(i)}.
$$

- **Full-batch gradient descent:** uses all $N$ examples per update; accurate
  but potentially expensive.
- **Stochastic gradient descent (SGD):** uses one example per update; cheap but
  noisy.
- **Mini-batch gradient descent:** uses a batch $\mathcal{B}$ of $B$ examples:

$$
\nabla_\theta J_{\mathcal B}(\theta)
=\frac{1}{B}\sum_{i\in\mathcal B}\nabla_\theta\mathcal{L}^{(i)}.
$$

Mini-batches balance computational efficiency with a less noisy gradient
estimate and map well onto GPU parallelism.

### 8. Generalization and regularization

The objective is not merely low training loss, but good performance on unseen
data.

- **Underfitting:** the model lacks enough capacity or has not learned the
  relevant structure.
- **Overfitting:** the model fits the training data but does not generalize.

Regularization constrains training to discourage unnecessarily complex or
brittle solutions.

#### Dropout

During training, dropout randomly suppresses activations. With dropout
probability $p$,

$$
\widetilde{\mathbf a}
=\frac{\mathbf m\odot\mathbf a}{1-p},
\qquad
m_j\sim\operatorname{Bernoulli}(1-p).
$$

The scaling preserves the expected activation. Dropout is disabled during
evaluation.

#### Early stopping

Track performance on a validation set and stop near the minimum validation
loss. Training loss may continue to decrease even after validation loss begins
to rise; that divergence is evidence of overfitting.

### 9. The complete training loop

For each batch:

1. **Forward pass:** compute predictions from the current parameters.
2. **Loss:** quantify prediction error.
3. **Backward pass:** compute gradients with backpropagation.
4. **Update:** modify parameters using an optimizer.
5. Repeat while monitoring both optimization and generalization.

The four central ideas to retain are therefore:

$$
\boxed{\text{representation} \;\longrightarrow\; \text{loss}
\;\longrightarrow\; \text{optimization} \;\longrightarrow\;
\text{generalization}}
$$

### 10. Connection to the Lecture 1 notebooks

| Notebook | Mathematical focus |
| --- | --- |
| `01-sigmoid-neuron-from-scratch.ipynb` | Affine map, sigmoid, BCE, analytical gradients, gradient descent |
| `02-sigmoid-neuron-pytorch.ipynb` | The same model expressed with tensors and automatic differentiation |
| `03-hidden-layer-network-from-scratch.ipynb` | Dense layers, nonlinearity, chain rule, and manual backpropagation |
| `04-hidden-layer-network-pytorch.ipynb` | The same hidden-layer network using PyTorch modules and optimizers |

## Questions and misconceptions

_Record unresolved questions and ideas that required correction._

## Exercises

_No exercises have been added yet._

## Review log

_Record later review dates, questions revisited, and remaining weak points._
