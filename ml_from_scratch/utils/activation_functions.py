import numpy as np
from numpy.typing import ArrayLike

def sigmoid(x: ArrayLike) -> np.ndarray:
    """
    Applies the sigmoid function element-wise.

    Formula: 1 / (1 + exp(-x))
    """
    return 1 / (1 + np.exp(-x))


def tanh(x: ArrayLike) -> np.ndarray:
    """
    Applies the hyperbolic tangent (tanh) function element-wise.

    Formula: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """
    return np.tanh(x)


def relu(x: ArrayLike) -> np.ndarray:
    """
    Applies the Rectified Linear Unit (ReLU) function element-wise.

    Formula: max(0, x)
    """
    return np.maximum(0, x)


def softmax(x: ArrayLike) -> np.ndarray:
    """
    Applies the softmax function to an array.

    Formula: exp(x) / sum(exp(x)) (along the last axis)
    """
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def leaky_relu(x: ArrayLike, alpha: float = 0.01) -> np.ndarray:
    """
    Applies the Leaky ReLU activation function element-wise.

    Formula: x if x > 0 else alpha * x
    """
    return np.where(x > 0, x, alpha * x)


def prelu(x: ArrayLike, alpha: float) -> np.ndarray:
    """
    Applies the Parametric ReLU (PReLU) activation function element-wise.

    Formula: x if x > 0 else alpha * x (where alpha is trainable)
    """
    return np.where(x > 0, x, alpha * x)


def elu(x: ArrayLike, alpha: float = 1.0) -> np.ndarray:
    """
    Applies the Exponential Linear Unit (ELU) activation function element-wise.

    Formula: x if x > 0 else alpha * (exp(x) - 1)
    """
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def gelu(x: ArrayLike) -> np.ndarray:
    """
    Applies the Gaussian Error Linear Unit (GELU) activation function element-wise.

    Formula: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def linear(x: ArrayLike) -> np.ndarray:
    """
    Applies a linear activation function.

    Formula: x (identity function)
    """
    return x
