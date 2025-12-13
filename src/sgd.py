import numpy as np
from typing import Callable, List
from mytypes import Array


def sgd(
    grad_f: Callable[[Array], Array],
    x0: Array,
    T: int,
    alpha: Callable[[int], float],
) -> List[Array]:
    """
    Stochastic Gradient Descent (SGD).

    Parameters
    ----------
    grad_f : Callable[[Array], Array]
        Function returning the gradient at a point x.
    x0 : Array
        Initial point.
    T : int
        Number of iterations.
    alpha : Callable[[int], float]
        Function defining the learning rate.

    Returns
    -------
    x_vals: List[Array]
        Trajectory of iterates x_k.
    """

    x: Array = np.asarray(x0, dtype=np.float64)
    x_vals: List[Array] = [x.copy()]

    for k in range(T):
        step: float = alpha(k)
        grad: Array = grad_f(x)
        x = x - step * grad
        x_vals.append(x.copy())

    return x_vals
