import numpy as np
from typing import Callable, List
from mytypes import Array


def polyak_standard_params(L, mu):
    """
    Compute standard Polyak heavy-ball parameters.
    """
    alpha = 4.0 / ((np.sqrt(L) + np.sqrt(mu)) ** 2)
    beta = ((np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))) ** 2
    return alpha, beta


def heavy_ball(
    grad_f: Callable[[Array], Array],
    x0: Array,
    T: int,
    alpha: float,
    beta: float,
) -> List[Array]:
    """
    Polyak's heavy ball method.

    Parameters
    ----------
    grad_f : Callable[[Array], Array]
        Function returning the gradient.
    x0 : Array
        Initial point.
    T : int
        Number of iterations.
    alpha : float
        Learning rate.
    beta : float
        Momentum parameter.

    Returns
    -------
    List[Array]
        Trajectory of iterates x_k.
    """

    x_prev: Array = np.asarray(x0, dtype=np.float64)
    x: Array = np.asarray(x0, dtype=np.float64)

    x_vals: List[Array] = [x.copy()]

    for _ in range(T):
        grad: Array = grad_f(x)
        x_new: Array = x - alpha * grad + beta * (x - x_prev)

        x_prev = x
        x = x_new
        x_vals.append(x.copy())

    return x_vals
