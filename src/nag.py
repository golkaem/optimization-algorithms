import numpy as np
from typing import Callable, List
from mytypes import Array


def nag(
    grad_f: Callable[[Array], Array],
    x0: Array,
    T: int,
    alpha: Callable[[int], float],
    beta: Callable[[int], float],
) -> List[Array]:
    """
    Nesterov's Accelerated Gradient (NAG) method.

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
    beta : Callable[[int], float]
        Function defining the momentum.

    Returns
    -------
    x_vals: List[Array]
        Trajectory of iterates x_k.
    """

    x: Array = np.asarray(x0, dtype=np.float64)
    v: Array = np.zeros_like(x, dtype=np.float64)
    x_vals: List[Array] = [x.copy()]

    for k in range(T):
        beta_k: float = beta(k)
        alpha_k: float = alpha(k)

        y: Array = x + beta_k * v
        grad_y: Array = grad_f(y)
        v = beta_k * v - alpha_k * grad_y
        x = x + v

        x_vals.append(x.copy())

    return x_vals
