import numpy as np
from typing import Callable, List
from hb import heavy_ball
from mytypes import Array


def cycle_score(
    x0: Array,
    grad_f: Callable[[Array], Array],
    alpha: float = 0.25,
    beta: float = 0.1,
    num_iters: int = 500,
    cycle_len: int = 2,
) -> float:
    """
    Evaluate how closely the trajectory produced by Polyak's heavy ball method
    resembles a periodic cycle of length cycle_len.

    The score is constructed so that smaller values correspond to trajectories
    that are closer to a stable limit cycle of the specified length.
    The function penalizes:
    (1) lack of compactness within each cycle phase,
    (2) insufficient separation between different phases,
    and (3) violations of temporal periodicity x_t ≈ x_{t - cycle_len}.

    Parameters
    ----------
    x0 : np.ndarray
        Initial point.
    grad_f : Callable[[np.ndarray], np.ndarray]
        Gradient of the loss function.
    alpha : float
        Learning rate of the heavy ball method.
    beta : float
        Momentum parameter of the heavy ball method.
    num_iters : int
        Number of heavy ball iterations.
    cycle_len : int
        Length of the cycle to be detected.

    Returns
    -------
    float
        Scalar score measuring similarity to a cycle of given length.
        Smaller values indicate stronger cycle-like behavior.
    """

    traj: List[Array] = heavy_ball(
        grad_f=grad_f, x0=x0, T=num_iters, alpha=alpha, beta=beta
    )

    # Берем последние K итераций
    K: int = 100
    tail: Array = np.asarray(traj[-K:], dtype=np.float64)
    T_tail: int = tail.shape[0]

    k: int = cycle_len

    # Разбиение траектории на k частей
    phases: List[List[Array]] = [[] for _ in range(k)]
    for idx, x in enumerate(tail):
        phase_id: int = idx % k
        phases[phase_id].append(x)

    phases_arr: List[Array] = [np.asarray(p, dtype=np.float64) for p in phases]

    # Центроиды
    mus: List[Array] = [np.mean(p, axis=0) for p in phases_arr]

    # Штраф за некомпактность
    comp_pen: float = 0.0
    for p, mu in zip(phases_arr, mus):
        comp_pen += float(np.mean(np.linalg.norm(p - mu, axis=1)))

    # Штраф за малую разделённость
    sep_pen: float = 0.0
    eps: float = 1e-8
    for i in range(k):
        for j in range(i + 1, k):
            dist: float = float(np.linalg.norm(mus[i] - mus[j]))
            sep_pen += 1.0 / (dist + eps)

    # Штраф за непериодичность
    period_pen: float = 0.0
    count: int = 0
    for t in range(k, T_tail):
        period_pen += float(np.linalg.norm(tail[t] - tail[t - k]))
        count += 1
    period_pen /= count

    return comp_pen + 0.01 * sep_pen + period_pen


def de_objective(
    grad_f: Callable[[Array], Array],
    alpha: float,
    beta: float,
    num_iters: int,
    cycle_len: int,
) -> Callable[[np.ndarray], float]:
    """
    Create an objective function compatible with scipy's differential evolution.

    Returns
    -------
    Callable[[np.ndarray], float]
        A function mapping an initial point x0 to a scalar fitness value.
    """

    def objective(x: np.ndarray) -> float:
        x0: Array = np.asarray(x, dtype=np.float64)
        return cycle_score(
            x0=x0,
            grad_f=grad_f,
            alpha=alpha,
            beta=beta,
            num_iters=num_iters,
            cycle_len=cycle_len,
        )

    return objective
