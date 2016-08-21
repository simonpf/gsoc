---
title: Deep Neural Networks in TMVA
author: Simon Pfreundschuh
date: 2016-06-14
---

This report documents the implementation of deep neural networks in TMVA.
After a brief description of the neural network model and the algorithms
involved in the training and evaluation of these networks, the design of
the implementation is outlined. The implementation currently consists of
two abstraction layers, a low-level interface and an object oriented model
neural network model. The low-level interface describes the numerically
demanding tasks, that are critical for the performance of the implementation
and will be optimized for the underlying computational architecture. The
object-oriented neural network model will be used to implement the actual
training of the network and will make use of the low-level interface.

# Mathematical Notation

Through this document, scalars will be written in regular, vectors in bold
($\mathbf{u}$) and matrices in capital bold ($\mathbf{W}$). Superscripts
denote layer indices, as in $\mathbf{u}^l$ for the activation vector
of the $l$th layer and subscript indices denote tensor indices as in $W^l_{i,j}$
for the weights of the $l$th layers in tensor representation.

# The Neural Network Model

For this specific implementation we restrict ourselves to feedforward neural
networks, where the activations $\mathbf{u}^l \in \mathbb{R}^{n_l}$ of
a layer $l$ are computed from the activations of the previous layer
using

\begin{align} \mathbf{u}^l =
f^l\left(\mathbf{W}^l\mathbf{u}^{l-1} + \boldsymbol{\theta}^l \right)
\end{align}

where $\mathbf{W}^l \in \mathbb{R}^{n_{l}\times n_{l-1}}$
and $\boldsymbol{\theta}^l \in \mathbb{R}^{n_l}$ are the weights and
bias values of layer $l$ and $f^l$ the corresponding *activation
function*. The activation function $f$ is a scalar function, which is
extended to vector or tensor arguments by element-wise application. The
network is assumed to consist of $n_h$ (hidden) layers.


## Input Data
The input data is assumed to be given by a set of samples or events,
represented by $n$-dimensional vectors $\mathbf{x} \in \mathbb{R}^n$.
A batch of input data consisting of $m$ events (samples) for the neural
network is given in matrix form by the matrix
$\mathbf{X} \in \mathbb{R}^{m \times n}$ whose rows are the input
events.


## Evaluation of the Network

For the evaluation of the network, first the activations $\mathbf{u}^1$
of the first layer with respect to the input vector
$\mathbf{x} \in \mathbb{R}^n$ are computed from

\begin{align}
    \mathbf{u}^1 &= f^1\left(\mathbf{W}\mathbf{x} + \boldsymbol{\theta^1}\right)
\end{align}

Then the remaining activations are propagated through the network according
to equation (1). The activation of the last layer $\mathbf{u}^{n_h}$ in
the network are then used to either evaluate a given *loss function* with
respect to a set of training labels $\mathbf{y}$ or produce a prediction
by applying a suitable transformation to the activations of the last layer.

# Methods and Algorithms

In this section the essential computational methods and corresponding
algorithms involved in the training and evaluation of the neural
network are presented. The two fundamental operations that we have to
perform on neural networks are the **forward propagation** of input values
through the net to obtain a prediction or evaluate the loss function as
well as the **backward propagation** of the gradients through the network
in order to compute the gradients required for the training of the
network.

## Neural Network Training

Our focus here lies on the training of neural networks, which can become very time
consuming for large data sets. A neural network may be viewed as a function mapping
a given input $\mathbf{x}$ onto a corresponding prediction $\hat{\mathbf{y}}$.

\begin{align}
\mathbf{x} \mapsto \hat{\mathbf{y}} \left (\mathbf{x}, \mathbf{W}, \boldsymbol{\theta} \right)
\end{align}

Apart form the architecture of the network, which we consider fixed, its output is
defined by the *network parameters* $\mathbf{W}, \boldsymbol{\theta}$, which
represent the weights and bias terms in each layer $l$ of the network. During
the training process a loss function
$J(\mathbf{\hat{Y}}\left(\mathbf{X}, W_{i,j}^l, \theta_{i,j}^l
\right))$ is minimized with respect to the network parameters over a given
training set consisting of input data
$\mathbf{X}$ and corresponding expected predictions $\mathbf{Y}$.
Note that the different notation for $\mathbf{x}$ and $\mathbf{X}$ as well as
$\mathbf{\hat{y}}$ and $\mathbf{\hat{Y}}$ has been used to distinguish the single
vector input and output ($\mathbf{x},\mathbf{\hat{y}}$) from the input and output
over the complete training set ($\mathbf{X}, \mathbf{\hat{Y}}$).


### Stochastic Gradient Descent

Stochastic gradient descent (SGD) is the most basic training method for deep neural
networks over large training sets. The key feature of *stochastic* gradient descent
as opposed to standard gradient descent is that each training step is performed on
a *mini batch* consisting only of a small number of samples from the training set.
In each step, the gradients $\frac{dJ}{dW^l_{i,j}},\frac{dJ}{\theta^l_{i,j}}$
of the loss function with respect to weight and bias terms of each layer are
computed using backpropagation. The weights and biases are then updated according to

\begin{align}
  W^l_{i,j} \to W^l_{i,j} - \alpha \frac{dJ}{dW^l_{i,j}},
  \theta^l_{i,j} \to \theta^l_{i,j} - \alpha \frac{dJ}{d\theta^l_{i,j}}
\end{align}

where $\alpha$ is the only parameter of the method, called the *learning rate*.


## Forward Propagation

For the forward propagation, we will consider the propagation of the
whole batch through the network. The input can then be viewed as as
two-dimensional tensor $x_{i,j}$ with the index $i$ running over the
training samples in the batch and the index $j$ running over the
features of each training sample. The neuron activations of each layer
then also take the form of two-dimensional tensors $u^l_{i,j}$,
with the index $i$ running over the training samples in the batch and
the index $j$ running over the neurons in the given layer. We will
refer to this tensor as the activation matrix of the given layer. The
activations of a given layer $u^n_{i,j}$ are then given in terms of
the activations of the previous layer $u^{n-1}_{i,j}$ by

\begin{align}
u^n_{i,j} &= f^n \left ( W^n_{j,k} u^{n-1}_{i,k} + \theta^{n}_j \right )
\end{align}

For increased readability we are using Einstein notation here, meaning
that repeated indices imply a sum over these indices with the
exception of indices that appear on the left-hand side of an
equation.The forward propagation of the neuron activations through the
network is illustrated in Figure 1.

![Forward propagation of activations through the network.](/home/simon/Documents/gsoc/blog/images/forward.png)

The computation of the activation matrices of each layer thus requires
the following operations:

* Computation of the tensor product $W^l_{j,k}u^l_{i,k}$
* Addition of the bias vectors $\theta^l_j$ along the second dimension (j) of the tensor
$W^l_{j,k}u^l_{i,k}$
* Application of the activation function $f^l$ to the tensor $W^l_{j,k}u^l_{i,k} + \theta^l_j$

## Backward Propagation

The gradients of the objective function with respect to the weights of
each layer are computed by repeated application of the chain rule of
calculus. Starting from the gradients of the objective function with
respect to the activations $u^{n_h}_{i,j}$ of the last layer in the
network $\frac{dJ}{du^{n_h}_{i,j}}$ the gradients can be computed from
\begin{align}
  \frac{dJ}{dW^l_{i,j}}      &= u^{l-1}_{m,j} (f^l)'_{m,i} \frac{dJ}{du^l_{m,i}} + R(W^l_{i,j}) \label{eq:bp1} \\
  \frac{dJ}{d\theta^l_{i}} &= (f^l)'_{m,i} \frac{dJ}{du^l_{m,i}} \\
  \frac{dJ}{du^{l-1}_{i,j}}  &= W^l_{n,j}(f^l)'_{i,n} \frac{dJ}{du^l_{i,n}} \label{eq:bp2}
\end{align}

Here, $R(W^n_{i,j})$ is an additional contribution to the gradient
from potential regularization terms in the objective function. The
term $(f^l)'_{m,i}$ is used to denote the first derivative of the
activation function evaluated at $W^l_{i,k}u^{l-1}_{m,k} +
\theta_i$. Note that the computations in equation (3) and (4) can be
implemented using element-wise as well as standard and transposed
matrix-matrix multiplication. The backpropagation algorithm is
illustrated in the figure below. Note that the contributions to the
gradients from regularization terms have been neglected in this
figure.

![Backward propagation of activations through the network.](/home/simon/Documents/gsoc/blog/images/backward.png)

The computation of the weight and bias gradients of a given layer thus
requires the following computations:

* Computation of the element-wise product of the derivatives of the
  activation function $(f^l_{i,j})'$ and the activation gradients of the
  layer and forward direction:
  \begin{align}
      t^l_{i,j} &= (f^l_{i,j})' \frac{dJ}{du^l_{i,j}}
  \end{align}
* Computation of the weight gradients $\frac{dJ}{dW^l_{i,j}}$ by computing
  the matrix product $u^{l-1}_{m,j}t^l_{m,i}$ of the transposed activation
  matrix of the previous layer and the matrix $t^l_{i,j}$.
* Computation of the bias gradients $\frac{dJ}{d\theta^l_{j}}$ by computing
  the column-wise sums of the matrix $t^l_{i,j}$
* Computation of the activation gradients of the previous layer
  $\frac{dJ}{du^{l-1}_{i,j}}$ by computing the matrix product of the matrix
  $t^l_{i,j}$ and the weight matrix $W^l_{i,j}$ of the current layer.

## Dropout

Droput, as introduced by Srivastava and Hinton [^1] has proven as a very effective
regularization technique that can significantly improve performance of deep neural
networks. The idea behind dropout regularization is to randomly
activate or deactivate the input activations to a given layer. The probability that
an input activation $u^{l-1}_{i,j}$ to layer $l$ is active is given by a Bernoulli
distribution with probability $p^l$. Since this effectively reduces the average input
activation to the layer, the resulting input activations should be scaled by the
factor $\frac{1}{p^l}$.

Applying dropout to a given layer thus amounts transforming the input activations
according to

\begin{align}
    \tilde{u}_{i,j}^{l-1} &= \frac{1}{p^l} r_{i,j}(p^l)\:u_{i,j}^{l-1} \\
    r_{i,j}(p^l) &= \begin{cases} 1 & \text{, with probability } p^l \\
                                  0 & \text{, with probability } 1 - l
                    \end{cases}
\end{align}



[^1]: [https://people.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)

# Low-level Interface

In this section the low-level interface is defined, which separates
the compute intensive, mathematical operations from the
general coordination of the training.

The low-level interface is implemented by architecture classes, that provide
an associated matrix type as well as the functions in the low-level interface
as static members.

```c++
class Architecture
{
   // Declare matrix and scalar type used.
   using Matrix_t = ...
   using Scalar_t = ...
   using DataLoader_t = ...

   // Low-level function declarations.
   ...
};
```

## Propagation
### Forward Propagation

We split the forward propagation of the activations through a given layer into
two steps. In the first step, the linear activations are computed using

\begin{align}
  \mathbf{t}^l &= \mathbf{u}^{l-1} \mathbf{W}^T + \boldsymbol{\theta}^l
\end{align}

Note that this is just the linear part of equation (3) but here written directly
as a matrix product using matrix notation. These operations are implemented by
the `MultiplyTranspose` and the `AddRowWise` functions in the low-level
interface.

```c++
static void MultiplyTranspose(Matrix_t &output,
                              const Matrix_t &input,
                              const Matrix_t &weights)
void AddRowWise(Matrix_t &output,
                const Matrix_t &biases)
```
In the second step the activation functions are applied
to the intermediate results produced in the first step. This is
handled by the generic `evaluate` function, which calls the
corresponding function of the low-level interface that implements
the application of the function represented by the given `EActivationFunction`
object to the given matrix. For more details on the representation of activation
functions, see the section on activation functions below.


```c++
template<typename Architecture_t>
inline void evaluate(typename Architecture_t::Matrix_t &A,
                     EActivationFunction f)

```

In addition to the neuron activations, also the first derivatives of the
activation functions are computed during the forward propagation phase.
Those are needed for the computation of the weight and bias gradients
during backpropagation. This is handled by the `evaluateDerivative`
function template, which similar to the `evaluate` function forwards the
call for the given activation function to the corresponding function
in the low-level interface.

```c++
template<typename Architecture_t>
inline void evaluateDerivative(typename Architecture_t::Matrix_t & B,
                               EActivationFunction f,
                               const typename Architecture_t::Matrix_t & A)
```

### Backward Propagation

Backward propagation is implemented by a single method in the
low-level interface in order to provide more flexibility to the
low-level implementation. For a given layer $l$, this function takes the
gradients of the objective function with respect to the activations of
the current layer (`activation_gradients`), the weights of the current
layer (`weights`) and the activations of the $l-1$ layer, i.e. the
previous layer (`activations_backward`). It also takes as input the first
derivatives of the activation functions (`df`), which should for efficiency be
computed during the forward propagation phase. From this the `Backward`
method computes the gradients of the objective function with respect to the
activations of the previous layer (`activation_gradients_backward`), the weights
of the current layer (`weight_gradients`) as well as the biases (`bias_gradients`)
of the current layer.

````c++
static void Backward(Matrix_t & activation_gradients_backward,
                     Matrix_t & weight_gradients,
                     Matrix_t & bias_gradients,
                     Matrix_t & df,
                     const Matrix_t & activation_gradients,
                     const Matrix_t & weights,
                     const Matrix_t & activations_backward,
                     Regularization r)
````
The primitive operations in the backward propagation are hadamard product,
matrix-matrix and transposed-matrix-matrix multiplication as well as a sum
over the columns of a matrix.

## Matrix Functions

All remaining low-level functionality required to implement neural networks
are functions or functionals acting on matrices. We group these functions
according to their tasks in the neural net:

- **Activation Functions**:  An activation or transfer function $f^l$ is applied to
the linear combination of the input activations of a given layer $l$ in order to
introduce non-linearities into the network.
- **Loss Functions**: Loss functions are functionals which quantify the performance
  of a neural network given the activation of the output layer and the true
  classification or regression results, e.g. *mean squared error* or
  *cross entropy*.
- **Output Functions**: Output functions transform the activation of the output
  layer in the network to a valid prediction, such as the *sigmoid transformation*
  or *softmax*.
- **Regularization**: Regularization functionals are applied to the weight matrices
  in the network and the scaled result is added to the network loss. Examples are
  L2 and L1 regularization.
- **Initialization**: Functions to initialize the weight matrices before the training.

The matrix functions are represented in the high-level implementation by enum types.
To avoid the duplication of boilerplate code, generic evaluation functions are used
in the high-level implementation that identify the correct function to be called from
the low-level interface, which is matched to a concrete implementation by overloading
on the corresponding matrix type.


### Activation Functions

The activation functions are represented by the `EActivationFunction` enum
class.

````c++
    /*! Enum that represents layer activation functions. */
    enum class EActivationFunction
    {
        IDENTITY = 'I',
        RELU     = 'R',
        SIGMOID  = 'S'
    };
````

The `evaluate` and `evaluateDerivative` function templates use a simple
`switch` expression to forward the call to the corresponding static function of
the corresponding architecture class. As naming convention we use the function
with only the first letter capitalized, which is suffixed by `Derivative`
for the computation of the first derivatives. As an example the signatures
for the evaluation of the ReLU function and its first derivative are given below:

```c++
static void Relu(Matrix_t &A);

static void ReluDerivative(Matrix_t &B, const Matrix_t &A);
```

### Loss Functions

For loss functions the approach is similar. They are again represented using
an `enum` class:

```c++
enum class ELossFunction
{
    CROSSENTROPY     = 'C',
    MEANSQUAREDERROR = 'R'
};
```

The difference with loss functions as opposed to the activation
functions is that they require two input matrices: The activation of
the output layer (`output`) in the net and the expected (true) training
predicitons (`Y`). Also, their first derivative takes the form of a
gradient. The high-level function templates for the evaluation of a
loss function and the computation of its gradients are thus given by

```c++
template<typename Matrix_t>
inline double evaluate(ELossFunction f,
                       const Matrix_t & Y,
                       const Matrix_t & OutputActivations)

template<typename Matrix_t>
inline void evaluateGradients(Matrix_t & dY,
                              ELossFunction f,
                              const Matrix_t &Y,
                              const Matrix_t &OutputActivations)
```


For the functions in the low-level interface implementing specific loss functions,
the same naming convention as for activation functions is adopted: The name of the
function in lower case letters which is suffixed by `_gradients` for the computation
of the gradients.


```c++
static Scalar_t MeanSquaredError(const Matrix_t &Y,
                                 const Matrix_t &output)
static void MeanSquaredError(Matrix_t & dY,
                             Matrix_t & Y,
                             Matrix_t & output)
```

### Output Functions

The output function of a neural network defines how a prediction is
obtained from the activations $u^{n_h}_{i,j}$ of the last layer in the
 network. Similar to the activation and loss functions, they are
represented by the `EOutputFunction` enum class.

```c++
enum class EOutputFunction
{
    SIGMOID  = 'S'
};
```

For the output functions only the evaluation of the functions is required.
However, the call to the evaluation function is slightly different since
for the output function one generally wants the output matrix to be different
from the input matrix. The signature for the function that evaluates the sigmoid
function for activations of the output layer `A` and writes the results into the
matrix `B` is given below.

```c++
static void Sigmoid(Matrix_t & B,
                    const Matrix_t & A)

```

### Regularization

For the treatment of regularization, we proceed in a similar way as above. The type
of the regularization is represented by an enum class

```c++
enum class ERegularization
{
    NONE = '0',
    L1   = '1',
    L2   = '2'
};
```

The generic `regularization` function computes the contribution of the
regularization to the loss of a given network prediction. The generic
method resolves the type of the regularization and forwards the call
to the corresponding device specific method.

```c++
template<typename Matrix_t>
inline auto regularization(const Matrix_t & A,
                           Regularization R)
    -> decltype(l1_regularization(A));
```

In addition to the computation of the contribution of the
regularization to the loss of the network, we need to add the gradient
of the regularization to the gradients of loss function
$\frac{dJ}{dW^l_{i,j}}$ with respect to the weights of each
layer. This is implemented by the `add_regularization_gradient`
method.

```c++
template<typename Matrix_t>
inline void addRegularizationGradient(Matrix_t &A,
                                      const Matrix_t &W,
                                      Regularization R)
```

The type signatures of the device specific functions that compute the L2
regularization for a given weight matrix and add the gradients of the L2
regularization term with respect to a given matrix to another matrix are
given below.

```c++
static RealType L2Regularization(const Matrix_t & W)
static void     AddL2RegularizationGradient(Matrix_t & A,
                                            Matrix_t & W)
```

And similarly for L1 regularization:

```c++
inline RealType L1Regularization(const Matrix_t & W)
inline void     AddL1RegularizationGradient(Matrix_t & A,
                                            const Matrix_t & W)
```

### Initialization

The initialization of the layers is treated in a similar way as the
other functions above. An enum class is used to represent the
initialization methdo and a generic function then forwards the call to
the `initialize` function to a device-specific implementation of the
desired initialization method.

```c++
enum class EInitializationMethod
{
    GAUSS    = 'G',
    UNIFORM  = 'U',
    IDENTITY = 'I',
    ZERO     = 'Z'
};
```

The function signatures for the device-specific initialization methods are given
below:

```c++
inline void InitializeGauss(Matrix_t & A);
inline void InitializeUniform(Matrix_t & A);
inline void InitializeIdentity(Matrix_t & A);
```
In addition to that a function is required that initializes the bias vector to
zero:

```c++
inline void InitializeZero(Matrix_t & A);
```

### Dropout

The dropout is performed by a single method in the low-level interface. This function
takes the probability $p^l$ that a given activation is active and randomly activates
or deactivates inputs to the layer and scales them by the reciprocal of the dropout
probability.

```c++
static void Dropout(Matrix_t A, Real_t probability);
```

### Data Loaders

Whith the functions defined above, the low-level interface provides all
functionality required for the training of neural networks. However, for
the training on accelerator additional factors must be considered. In general,
accelerator devices have only a limited amount of on board memory that **cannot**
be accessed directly from the host machine. The low-level interface must therefore
provide sufficient flexibility for the backend implementations to manage their
transfer to and from the device.

To handle this, each implementation of the low-level interface must provide
an associated data loader class that takes care of preparing the batches as input
to the neural network for training and testing of the network.

```c++
class Architecture
{
   ...
   using DataLoader_t = ...
   ...
};
```

Each data loader implementation should provide `begin()` and `end()` routines
returning iterators to the first and the last batch in the current epoch. Using
the batch iterator object, the training implementation can then loop over the
batches to train the network. The dataloader should also take care of the
shuffling of the data.

```c++
class DataLoader
{
   ...
   BatchIterator begin();
   BatchIterator end();
   ...
};
```

A single batch basically just groups together the matrix representation for the
input samples and output classes for the neural net. Access to those should be
provided by `GetInput()`, `GetOutput()` methods:

```c++
class DataLoader
{
   ...
   BatchIterator begin();
   BatchIterator end();
   ...
};
```

# The Object-Oriented Neural Network Model

The object-oriented neural network model is the high-level implementation of
the actual neural network and makes use of the low-level interface presented
above.

## Architecture Independence

Independence of the implementation from the underlying hardware
architecture is achieved through the use of template programming. Each
of the classes in the implementation takes an architecture class as
template argument, which provides public `Real_t` and `Matrix_t` type
aliases which define the types used for the representation of scalars and
Matrices. In addition to that the each architecture class must
provide static functions that implement the low-level interface described
above.

## The Neural Network Model

The neural network implementation itself consists of two main components:
The `TNet` and the `TLayer` class templates that represent a complete
neural network containing a collection of layers and a single layer of such
a network, respectively.
In addition to the standard `TLayer` class there is also a `TSharedLayer` class
that can also be used as a layer type for the `TNet` class. The difference between
the `TLayer` and the `TSharedLayer` classes is that the `TSharedLayer` class only
holds references to weight and bias matrices of another layer. This can be used
for example to create a network with a different batch size for the evaluation
on a validation or test set or to do multi-threaded training in hogwild style
(see below).

![Class structure of the object-oriented neural network model](classes.png)

### The `TNet` Class

```c++
template<typename Architecture_t, typename Layer_t>
class TNet;
```

The `TNet` class represents a concrete neural network through a `std::vector`
of the layer type `Layer_t` provided as template argument. In addition
to that a `TNet` object also holds the loss function `fJ` of the layer
that is minimized during training and the type of regularization `fR`
that is applied to the network.

The neural network handles all memory that is required for the
evaluation of the network and the computation of its gradients. Since
this memory is dependent on the batch size of the input, a particular `TNet`
object has a specific, associated batch size, which is represented by
the member variable `fBatchSize`. An identical network for a different
batch size can be created using the clone method (see below). The
number of features of an input event is represented by the
`inputWidth` member variable, as it corresponds to the width of an
additional *input layer* of the network.
The general class interface is illustrated in the figure below.

![Interface of the `Net` class](class_net.png)

The interface provided by the `TNet` class should be mostly self-explanatory.
A detailed description of each function can be found in the source code or
the corresponding doxygen documentation.

#### Caveats

Currently the implementation of the backpropagation modifies temporary values that
are computed during the forward propagation. The call to the `Backward(...)` function
must therefore occur directly after the corresponding call to the `Forward(...)`
method.

### The `TLayer` Class

The layer class represents a basic layer of the network and manages the memory
required for the foward propagation of activation and backward propagation
of gradients through this given layer. Each layer has as attributes a given width
$n_l$ (`fWidth`), which is the number of neurons in this layer, and an activation
function `f` (`fF`).

On creation, each layer allocates memory for all matrices that have to be
computed during the forward and backward propagation steps to hold neuron
activations, gradients and temporary values. For a given batch size $n_b$
those are:

* The $n_l \times n_{l-1}$ weight matrix `weights`
* The $n_l \times 1$ bias matrix `biases`
* The $n_b \times n_l$ activation matrix `output` (computed during forward propagation)
* The $n_b \times n_l$ matrix $(f^l_{i,j})'$ matrix of first derivatives of
  the activation function `derivatives` (computed during forward propagation)
* The $n_b \times n_l$ matrix containing the gradients of the loss function with
  respect to the activations of this layer `activation_gradients`
* The $n_l \times n_{l-1}$ matrix containing the gradients of the loss function
  with respect to the activations of this layer.
* The $n_l \times 1$ matrix containing the gradients of loss function with respect
  to the bias values of this layer.

The `TLayer` class provides the `Forward` and `Backward` member functions that
propagate neuron activations forward and gradients backward through
the layer. The general interface of the layer class is illustrated in
the figure below.

![Structure of the `TLayer` class](class_layer.png)

### The `SharedLayer` Class

The `TSharedLayer` class is mostly identical to the `TLayer` class
except for that it does not have its own weight and bias matrices but
only holds references to the weights of another layer. This is
required in order to evaluate networks on different batch size, but
also for the implementation of multithreaded training using *Hogwild!*
[^2] style.

[^2]: [https://people.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf](https://people.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf)

# Profiling

In order to obtain a realistic analysis of the performance of the different
implementations, we also define a general profiling interface for the methods
in the backend. This will, in addition to architecture specific profiling,
provide a convenient way of comparing the different backends.

## Floating point operations

In order to assess the performance of the implementation, it is helpful to
compare the numerical throughput to the theoretical peak performance of the
corresponding computing architecture. For the computation of the throughput
the number of floating point operations required for forward and backward
propagation through the network are required.

### Forward Propagation

Consider a layer with $n^l$ neurons and a batch size $n_b$. For the forward
propagation of the input activations $\mathbf{u}^{l-1}$ through this layer
the following operations have to be performed:

- Multiplication of the input activations with the transposed weight matrix
  $\mathbf{W}^l$
- Row-wise addition of the bias vector $\mathbf{\theta}^{l}$
- Application of the activation function

The matrix multiplication requires $n^l n_b (2 n^{l-1} - 1) \: FLOPs$. The
row-wise addition of the bias terms requires $n^l n_b \: FLOPs$. The arithmetic
intensity of the application function depends of course on the activation function.
In general the number of $FLOPs$ will be given by

\begin{align}
    n^l n_b n_f
\end{align}

for $n_f$ being the number of flops required for a single evaluation. Here the
following estimates for $n_f$ will be used:

| Operation  |  CPU  |  GPU  |
|:----------:|:-----:|:-----:|
| Add/Sub/Mul|   1   |  1    |
| Division   |   5   |  1    |
| Exp        |  10   |  1    |
| TanH       |  20   |  1    |


| Acitvation Function  |  CPU  |  GPU  |
|:--------------------:|:-----:|:-----:|
| Identity             |   0   |   0   |
| Relu                 |   1   |   1   |
| Sigmoid              |  17   |   4   |
| TanH                 |  20   |   1   |
| Symmetric Relu       |  1    |   1   |
| Soft Sign            |  8    |   4   |
| Gauss                |  12   |   3   |

| Acitvation Function  |  CPU  |  GPU  |
|:--------------------:|:-----:|:-----:|
| Identity             |   0   |   0   |
| Relu                 |   1   |   1   |
| Sigmoid              |  19   |   6   |
| TanH                 |  22   |   3   |
| Symmetric Relu       |  1    |   1   |
| Soft Sign            |  8    |   4   |
| Gauss                |  14   |   5   |
