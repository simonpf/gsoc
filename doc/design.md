---
title: Deep Neural Networks in TMVA
author: Simon Pfreundschuh
date: 2016-06-14
---

This report documents the implementation of deep neural networks in TMVA.
After a brief description of the neural network model and the algorithms
involved in the training and evaluation of such networks, the design of
the implementation is outlined. The implementation currently consists of
two abstraction layers, a low-level interface and an object oriented model
neural network model. The low-level interface describes the numerically
demanding tasks, that are critical for the performance of the implementation
and will be optimized for optimal performance. The object-oriented neural
network model will be used to implement the actual training of the network
and will make use of the low-level interface.

# The Neural Network Model

For this implementation we restrict ourselves to feedforward neural
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

Deep neural networks are trained by optimizing a loss function $J\left
( \mathbf{y},\mathbf{u}^{l_h},\mathbf{W} \right )$ that quantifies the
correctness of the network prediction corresponding to the activations
of the output layer $\mathbf{u}^{l_h}$ with respect to the true values
$\mathbf{y}$ and possibly also including regularization terms, which
are functions of the weights $\mathbf{W}$ of the network. This
objective function is then minimized by applying a gradient-based
optimization technique, usually a modification of the gradient descent
method. The key to scalable training of deep neural networks is the
training on small, randomized subsets of the training data, so called
*batches*. The gradient of the neural network loss $J$ is then computed for
a single batch and used as an approximation to the true gradient over
the whole training set to train the network.

## Forward Propagation

For the forward propagation, we will consider the propagation of the
whole batch through the network. The input can then be viewed as as
two-dimensional tensor $x_{i,j}$ with the index $i$ running over the
training samples in the batch and the index $j$ running over the
features of each training sample. The neuron activations of each layer
can then also be represented by two-dimensional tensors $u^l_{i,j}$,
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

The computation of the weight and bias gradients of a give layers thus
require the following computations:

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

# Low-level Interface

In this section the low-level interface is presented, which will
separate the compute intensive, mathematical operations from the
general coordination of the training.

## Forward Propagation

We split the forward propagation of the activations through a given layer into
two steps. In the first step, the linear activations are computed using

\begin{align}
  \mathbf{t}^l &= \mathbf{u}^{l-1} \mathbf{W}^T + \boldsymbol{\theta}^l
\end{align}

Note that this is just the linear part of equation (3) but here written directly
as a matrix product using matrix notation. These operations are implemented by
the `multiply_transpose` and the `add_row_wise` functions in the low-level
interface.

```c++
void multiply_transpose(&output,
                        const MatrixType &input,
                        const MatrixType &weights)

void add_row_wise(MatrixType &output,
                  const MatrixType &biases)
```

## Backward Propagation

The backward propagation is implemented by a single method in the
low-level interface. For a given layer $l$, this function takes the
gradients of the objective function with respect to the activations of
the current layer (`activation_gradients`), the weights of the current
layer (`weights`) and the activations of the $l-1$ layer, i.e. the
previous layer (`activations_backward`). It also takes as input the first
derivatives of the activation functions, which should ideally be
computed during the forward propagation phase. From this the `backward`
method computes the gradients of the objective function with respect to the
activations of the previous layer, the weights of the current layer as well as
the biases of the current layer.

Note that the formulas for backpropagation can be implemented using
element-wise matrix multiplication as well as standard and transposed
matrix-matrix multiplication.

````c++
template<typename MatrixType>
void backward(MatrixType & activation_gradients_backward,
                MatrixType & weight_gradients,
                MatrixType & bias_gradients,
                MatrixType & df,
                const MatrixType & activation_gradients,
                const MatrixType & weights,
                const MatrixType & activations_backward,
                Regularization r)
````

## Activation Functions

The activation functions are represented by the `ActivationFunction` enum
class.

````c++
        /*! Enum that represents layer activation functions. */
        enum class ActivationFunction
        {
            IDENTITY = 'I',
            RELU     = 'R',
            SIGMOID  = 'S'
        };
````

The evaluation of the activation functions is performed using the
`evaluate` function template which forwards the call to the
actual evaluation function corresponding to the given type of the
activation function.

```c++
// Apply the activation function f to the matrix A.
template<typename MatrixType>
inline void evaluate(MatrixType &A,
                     ActivationFunction f)

```

An architecture-specific implementation simply
overloads the evaluation function for the architecture-specific
matrix data type. As an example the signature for the evaluation of
the ReLU function is given below:

```c++
// Apply the ReLU functions to the elements in A.
inline void relu(MatrixType &A);
```

In addition to the evaluation of the activation functions, also a function that
 computes the first order derivatives of the activation functions must be provided.
 The signature for the computation of the first order derivatives of the ReLU
 function is also given below.

```c++
// For each element in A evaluate the first derivative of the ReLU function
// and write the result into B.
inline void relu_derivative(MatrixType &B, const MatrixType &A);
```

## Loss Functions

Similar to the activation functions, the loss functions are also represented by an
enum class.

```c++
enum class LossFunction
{
    CROSSENTROPY     = 'C',
    MEANSQUAREDERROR = 'R'
};
```

The evaluation of the loss functions is implemented by the overloaded
`evaluate` function, which computes the loss from a given activation
matrix of the output layer of the network and the matrix $\mathbf{Y}$
containing the true predictions. Similar to the implementation of
activation functions, the generic `evaluate` function forwards the
call to a given device-pecific evaluation function, that is overloaded
with the device-specific matrix type.

```c++
template<typename MatrixType>
inline double evaluate(LossFunction f,
                const MatrixType & Y,
                const MatrixType & output)
```

In addition to that, we need to be able to compute the gradient of the loss
function with respect to the activations of the output layer. This is
implemented by the `evaluate_gradient` function, which also just forwards
the call for a given loss function to the corresponding device-specific function.

```
template<typename MatrixType>
inline void evaluate_gradient(MatrixType & dY,
                        LossFunction f,
                        const MatrixType &Y,
                        const MatrixType &output)
```

The signature of a device-specific implementation of the that computes the
mean squared error for a concrete matrix type `MatrixType` is given below:

```c++
inline RealType mean_squared_error(const MatrixType &Y,
                                   const MatrixType &output)
```

## Output Functions

The output function of a neural network defines how a prediction is
obtained from the activations $u^{n_h}_{i,j}$ of the last layer in the
 network. Similar to the activation and loss function, they are
represented by the `OutputFunction` enum class.

```c++
/*! Enum that represents output functions */
enum class OutputFunction
{
    SIGMOID = 'S'
};
```

For the output functions only the evaluation of the functions is required.
However, the call to the evaluation function is slightly different since
for the output function one generally wants the output matrix to be different
from the input matrix. The signature for the function that evaluates the sigmoid
function for activations of the output layer `A` and writes the results into the
matrix `B` is given below.

```c++
// Apply the sigmoid function to the elements in A and write the
// results into B.
template<typename RealType>
inline void sigmoid(MatrixType & B,
                    const MatrixType & A)

```

## Regularization

For the treatment of regularization, we proceed in a similar way as above. The type
of the regularization is represented by an enum class

```c++
/*! Enum representing the regularization type applied for a given layer */
enum class Regularization
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
template<typename MatrixType>
    auto regularization(const MatrixType & A,
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
template<typename MatrixType>
    void add_regularization_gradient(MatrixType &A,
                                     const MatrixType &W,
                                     Regularization R)
```

The type signatures of the device specific functions that compute the L2
regularization for a given weight matrix and add the gradients of the L2
regularization term with respect to a given matrix to another matrix are
given below.

```c++
inline RealType l2_regularization(const MatrixType & W)
inline void     add_l2_regularization_gradient(MatrixType & A,
                                               MatrixType & W)
```

And similarly for L1 regularization:

```c++
inline RealType l1_regularization(const MatrixType & W)
inline void     add_l1_regularization_gradient(MatrixType & A,
                                               const MatrixType & W)
```

## Initialization

The initialization of the layers is treated in a similar way as the
other functions above. An enum class specifies which type of
initialization should be performed and the and a generic function then
forwards the call to the `initialize` function to a device-specific
implementation of the desired initialization method.

```c++
/* Enum represnting the initialization method used for this layer. */
enum class InitializationMethod
{
    GAUSS    = 'G',
    UNIFORM  = 'U',
    IDENTITY = 'I'
};
```

The function signatures for the device-specific initialization methods are given
below:

```c++
inline void initialize_gauss(MatrixType & A);
inline void initialize_uniform(MatrixType & A);
inline void initialize_identity(MatrixType & A);
```

# The Object-Oriented Neural Network Model

The object-oriented neural network model is the high-level implementation of
the actual neural network and makes use of the low-level interface presented
above.

## Architecture Independence

Independence of the implementation from the underlying hardware
architecture is achieved through the use of template programming. Each
of the classes in the implementation takes an `Architecture` template
argument, which provides public types `RealType` and `MatrixType` that
define the number type used for the representation of scalars as well
as the type used for the representation of matrices.

## Implementation Model

The implementation model consists of two main components: two classes
representing layers of the network and a class representing the neural
network, which consists of an arbitrary number of layers. Currently
two different layer classes are available, the basic `Layer` class and
the `SharedLayer` class for layers that share weight and bias
matrices. Polymorphism with respect to the layer class is also
achieved through the use of template programming. The general class
structure is illustrated in the Figure below.

![Class structure of the object-oriented neural network model](classes.png)

### The `Net` Class

```c++
template<typename Architecture, typename LayerType>
class Net;
```

The `Net` class represents a concrete neural network through a `std::vector`
of layers of type `LayerType`. In addition to that a `Net` object also holds the
loss function `J` of the layer that is minimized during training and the type
of regularization `R` that is applied to the network.

The neural network handles all memory that is required for the
evaluation of the network and the computation of its gradients. Since
this memory is dependend on the batch size of the input, a `Net`
object has a specific, associated batch size, which is represented by
the member variable `batch_size`. An identical network for a different
batch size can be created using the clone method (see below). The
number of features of an input event is represented by the
`input_widht` member variable, as it corresponds to the width of an
additional *input layer* of the network.
The general class interface is illustrated in the figure below.

![Interface of the `Net` class](class_net.png)

The interface provided by the `Net` class should be mostly self-explanatory.
A detailed description of each function can be found in the source code or
the corresponding doxygen documentation.

#### Caveats


Currently the implementation of the backpropagation modifies temporary values that
are computed during the forward propagation. The call to the `backward(...)` function
must therefore occur directly after the corresponding call to the `forward(...)`
method.

### The `Layer` Class

The layer class represents a basic layer of the network and manages the memory
required for the foward propagation of activation and backward propagation
of gradients through the network. Each layer has a given width $n_l$, which is the
number of neurons in this layer, and an activation function `f`.

On creation, each layer allocates memory for all matrices that have to be
computed during the forward and backward propagation steps to hold neuron
activations, gradients and temporary values. For given batch size $n_b$
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

The `Layer` class provides the `forward` and `backward` methods that propagate
neuron activations forward and gradients backward through the layer. The general
interface of the layer class is illustrated in the figure below.

![Interface of the `Layer` class](class_layer.png)

### The `SharedLayer` Class

The `SharedLayer` class is mostly identical to the `Layer` class
except for that it does not have its own weight and bias matrices but
only holds references to the weights of another layer. This is
required in order to evaluate networks on different batch size, but
also for the implementation of multithreaded training using *Hogwild!*
[^1] style.

[^1]: [https://people.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf](https://people.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf)


