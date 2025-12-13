import numpy as np
from typing import Callable, List
from mytypes import Array


def adam(
    grad_f: Callable[[Array], Array],
    x0: Array,
    T: int,
    alpha: Callable[[int], float],
    beta1: Callable[[int], float],
    beta2: Callable[[int], float],
    epsilon: float = 1e-8,
) -> List[Array]:
    """
    Adaptive Moment Estimation (Adam) optimizer.

    Parameters
    ----------
    grad_f : Callable[[Array], Array]
        Function returning the gradient at point x.
    x0 : Array
        Initial point.
    T : int
        Number of iterations.
    alpha : Callable[[int], float]
        Function defining the learning rate.
    beta1 : Callable[[int], float]
        Function defining the exponential decay rate for the first moment.
    beta2 : Callable[[int], float]
        Function defining the exponential decay rate for the second moment.
    epsilon : float
        Small constant value.

    Returns
    -------
    x_vals: List[Array]
        Trajectory of iterates x_k.
    """

    # Инициализация
    x: Array = np.asarray(x0, dtype=np.float64)
    m: Array = np.zeros_like(x, dtype=np.float64)
    v: Array = np.zeros_like(x, dtype=np.float64)
    x_vals: List[Array] = [x.copy()]

    for k in range(1, T + 1):
        g: Array = grad_f(x)
        a: float = alpha(k)
        b1: float = beta1(k)
        b2: float = beta2(k)

        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        m_hat: Array = m / float(1 - b1**k)
        v_hat: Array = v / float(1 - b2**k)

        x = x - a * m_hat / (np.sqrt(v_hat) + epsilon)

        x_vals.append(x.copy())

    return x_vals
