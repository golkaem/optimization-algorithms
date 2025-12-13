import matplotlib.pyplot as plt
from typing import List
from mytypes import Array


def show(
    x_vals: List[Array],
    f_vals: List[float],
    title1: str = r"Значения $x_k$",
    title2: str = r"Значения функции $f(x)$ в точках $x_k$",
    ylabel1: str = r"$x$",
    ylabel2: str = "",
    x_min: float = 0.0,
    f_min: float = 0.0,
) -> None:

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(
        0,
        x_vals[0],
        color="blue",
        linestyle="--",
        label=f"Начальная точка $x_0 = {x_vals[0][0]:.2f}$",
        zorder=5,
    )
    plt.axhline(
        x_min,
        color="red",
        linestyle="--",
        alpha=0.4,
        label=f"Минимум $x = {x_min:.2f}$",
    )
    plt.plot(x_vals)
    plt.legend()
    plt.xlabel("Итерации")
    plt.ylabel(ylabel1)
    plt.title(title1)
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(
        0,
        f_vals[0],
        color="blue",
        linestyle="--",
        label=f"$f(x_0) = {f_vals[0]:.2f}$",
        zorder=5,
    )
    plt.axhline(
        f_min,
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"Минимум $f(x) = {f_min:.2f}$",
    )
    plt.plot(f_vals)
    plt.legend()
    plt.xlabel("Итерации")
    plt.ylabel(ylabel2)
    plt.title(title2)
    plt.grid(True)

    plt.tight_layout()
    plt.show()
