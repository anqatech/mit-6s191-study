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

#### Worked derivation: the $2\rightarrow2\rightarrow1$ XOR network

The hidden-layer notebook stores the $N$ training examples as **rows**, rather
than deriving one example at a time as column vectors. Let the input dimension
be $d=2$ and the hidden width be $h=2$. The forward-pass quantities have these
shapes:

| Quantity | Shape |
| --- | --- |
| $\mathbf X$ | $(N,d)$ |
| $\mathbf W_1$ | $(d,h)$ |
| $\mathbf b_1$ | $(1,h)$ |
| $\mathbf Z_1,\mathbf A_1$ | $(N,h)$ |
| $\mathbf W_2$ | $(h,1)$ |
| $b_2$ | $(1,1)$ |
| $\mathbf Z_2,\hat{\mathbf y},\mathbf y$ | $(N,1)$ |

The forward pass is

$$
\mathbf Z_1=\mathbf X\mathbf W_1+\mathbf b_1,
\qquad
\mathbf A_1=\tanh(\mathbf Z_1),
$$

$$
\mathbf Z_2=\mathbf A_1\mathbf W_2+b_2,
\qquad
\hat{\mathbf y}=\sigma(\mathbf Z_2).
$$

The biases are broadcast across the $N$ rows. Define

$$
d\mathbf Q\equiv\frac{\partial J}{\partial\mathbf Q},
$$

so $d\mathbf Q$ always has the same shape as $\mathbf Q$.

##### Step 1: derivative at the output logits

For example $i$, the binary cross-entropy loss is

$$
\mathcal L_i
=-\left[
y_i\log(\hat y_i)+(1-y_i)\log(1-\hat y_i)
\right],
\qquad
\hat y_i=\sigma(z_{2,i}).
$$

Differentiate both parts of the composition:

$$
\frac{\partial\mathcal L_i}{\partial\hat y_i}
=-\frac{y_i}{\hat y_i}
 +\frac{1-y_i}{1-\hat y_i}
=\frac{\hat y_i-y_i}{\hat y_i(1-\hat y_i)},
$$

$$
\frac{\partial\hat y_i}{\partial z_{2,i}}
=\hat y_i(1-\hat y_i).
$$

The chain rule cancels the probability factors:

$$
\frac{\partial\mathcal L_i}{\partial z_{2,i}}
=\frac{\partial\mathcal L_i}{\partial\hat y_i}
 \frac{\partial\hat y_i}{\partial z_{2,i}}
=\hat y_i-y_i.
$$

Because the empirical loss is the mean

$$
J=\frac{1}{N}\sum_{i=1}^{N}\mathcal L_i,
$$

the batch error at the output logits is

$$
\boxed{
d\mathbf Z_2=\frac{1}{N}(\hat{\mathbf y}-\mathbf y)
}
\qquad\text{with shape }(N,1).
$$

##### Step 2: gradients of the output-layer parameters

For one example,

$$
z_{2,i}=\sum_{j=1}^{h}A_{1,ij}W_{2,j}+b_2.
$$

For an output weight $W_{2,j}$,

$$
\frac{\partial z_{2,i}}{\partial W_{2,j}}=A_{1,ij}.
$$

Summing the contribution from every example gives

$$
\frac{\partial J}{\partial W_{2,j}}
=\sum_{i=1}^{N}
\frac{\partial J}{\partial z_{2,i}}
\frac{\partial z_{2,i}}{\partial W_{2,j}}
=\sum_{i=1}^{N}(dZ_2)_iA_{1,ij}.
$$

This is the matrix product

$$
\boxed{
d\mathbf W_2=\mathbf A_1^{\mathsf T}d\mathbf Z_2
}
\qquad\text{with shape }(h,1).
$$

Since $\partial z_{2,i}/\partial b_2=1$,

$$
\boxed{
db_2=\sum_{i=1}^{N}(dZ_2)_i
}
\qquad\text{with shape }(1,1).
$$

The sum is over examples only. The $1/N$ averaging factor is already contained
inside $d\mathbf Z_2$.

##### Step 3: propagate the error into the hidden activations

The hidden activation $A_{1,ij}$ affects only the corresponding example's
output logit, and

$$
\frac{\partial z_{2,i}}{\partial A_{1,ij}}=W_{2,j}.
$$

Therefore,

$$
(dA_1)_{ij}
=\frac{\partial J}{\partial A_{1,ij}}
=(dZ_2)_iW_{2,j}.
$$

Writing every example and hidden neuron simultaneously gives

$$
\boxed{
d\mathbf A_1=d\mathbf Z_2\mathbf W_2^{\mathsf T}
}
\qquad\text{with shape }(N,h).
$$

##### Step 4: propagate through the elementwise $\tanh$

For every example $i$ and hidden neuron $j$,

$$
A_{1,ij}=\tanh(Z_{1,ij}),
\qquad
\frac{\partial A_{1,ij}}{\partial Z_{1,ij}}
=1-A_{1,ij}^{2}.
$$

The componentwise chain rule is therefore

$$
\boxed{
(dZ_1)_{ij}
=(dA_1)_{ij}\left(1-A_{1,ij}^{2}\right)
}.
$$

There is no matrix square in this expression. If
$\mathbf 1_{N\times h}$ denotes an array of ones matching the hidden
activation's shape, the complete batch can be written as

$$
\boxed{
d\mathbf Z_1
=d\mathbf A_1\odot
\left(
\mathbf 1_{N\times h}-\mathbf A_1\odot\mathbf A_1
\right)
}
\qquad\text{with shape }(N,h),
$$

where $\odot$ means elementwise multiplication.

##### Step 5: gradients of the input-to-hidden parameters

Each hidden pre-activation is

$$
Z_{1,ij}
=\sum_{k=1}^{d}X_{ik}W_{1,kj}+b_{1,j}.
$$

For one weight $W_{1,kj}$,

$$
\frac{\partial Z_{1,ij}}{\partial W_{1,kj}}=X_{ik},
$$

so

$$
\frac{\partial J}{\partial W_{1,kj}}
=\sum_{i=1}^{N}(dZ_1)_{ij}X_{ik}.
$$

In matrix form,

$$
\boxed{
d\mathbf W_1=\mathbf X^{\mathsf T}d\mathbf Z_1
}
\qquad\text{with shape }(d,h).
$$

Because $\partial Z_{1,ij}/\partial b_{1,j}=1$,

$$
\boxed{
(d b_1)_j=\sum_{i=1}^{N}(dZ_1)_{ij}
}
\qquad\text{with shape }(1,h).
$$

Again, the sum is across examples, not across hidden neurons: each hidden
neuron has its own bias and therefore its own bias gradient.

##### Complete backward sequence

The complete computation is

$$
\boxed{
d\mathbf Z_2
\longrightarrow
\{d\mathbf W_2,db_2\}
\longrightarrow
d\mathbf A_1
\longrightarrow
d\mathbf Z_1
\longrightarrow
\{d\mathbf W_1,d\mathbf b_1\}
}.
$$

With the averaging factor placed in $d\mathbf Z_2$, none of the downstream
gradients should be divided by $N$ again.

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
