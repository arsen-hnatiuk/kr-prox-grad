import numpy as np
import sys
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Callable

module_path = Path(__file__).resolve().parent.parent
if module_path not in sys.path:
    sys.path.append(str(module_path))
src_path = (module_path / "src").resolve()
if src_path not in sys.path:
    sys.path.append(str(src_path))
from lib.measure import Measure
from kr_prox_grad import KR_PROX_GRAD
from frank_wolfe import FRANK_WOLFE

results_dir = Path("results/source_identification")

results_dir.mkdir(parents=True, exist_ok=True)

# Generate data and define functions

Omega = np.array([[0, 1], [0, 1]])
discretization_resolution = 100
beta = 1e-1
observation_resolution = 4
std_factor = 0.1
true_sources = np.array([[0.28, 0.71], [0.51, 0.27], [0.71, 0.53]])
true_weights = np.array([1, 0.7, 0.8])


def get_grid(size: int) -> np.ndarray:
    grid = (
        np.array(
            np.meshgrid(
                *(np.linspace(bound[0], bound[1], size + 2)[1:-1] for bound in Omega)
            )
        )
        .reshape(len(Omega), -1)
        .T
    )
    return grid


observations = get_grid(observation_resolution)
discretization_domain = get_grid(discretization_resolution)


def kernel(x):
    # Input is 2D array of shape (number of points, Omega dimension)
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    columns = []
    outer_factor = np.sqrt(std_factor * np.pi) ** Omega.shape[0]
    for point in observations:
        diff = point - x  # (len(x), Omega.shape[0])
        norms = -np.square(np.linalg.norm(diff, axis=1)) / std_factor  # (len(x),)
        exponentiated = np.exp(norms)  # (len(x),)
        columns.append(exponentiated)
    result = (
        np.transpose(np.array(columns), axes=(1, 0)) / outer_factor
    )  # shape=(len(x), len(observations))
    return result


def grad_kernel(x):
    # Input is 2D array of shape (number of points, Omega dimension)
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    gradients = []
    outer_factor = std_factor * np.sqrt(std_factor * np.pi) ** Omega.shape[0] / 2
    for point in observations:
        diff = point - x  # (len(x), Omega.shape[0])
        norms = -np.square(np.linalg.norm(diff, axis=1)) / std_factor  # (len(x),)
        exponentiated = np.exp(norms)  # (len(x),)
        gradient = diff * exponentiated.reshape(-1, 1)  # shape=(len(x),Omega.shape[0])
        gradients.append(gradient)
    result = (
        np.transpose(np.array(gradients), axes=(1, 0, 2)) / outer_factor
    )  # The Jacobian of kappa, shape=(len(x), len(observations), Omega.shape[0])
    return result


u_hat = Measure(support=true_sources, coefficients=true_weights)
target = u_hat.duality_pairing(kernel)

g = lambda u: beta * np.linalg.norm(u, ord=1)
f = lambda y: 0.5 * np.linalg.norm(y - target) ** 2
j = lambda u: f(u.duality_pairing(kernel)) + g(u.coefficients)

kernel_norm_0 = np.max(np.linalg.norm(kernel(discretization_domain), axis=1))
kernel_norm_1 = np.max(np.linalg.norm(grad_kernel(discretization_domain), axis=(1, 2)))
L = max(kernel_norm_0, kernel_norm_1) ** 2


def p(u):
    Ku = u.duality_pairing(kernel)
    inner = Ku - target
    return lambda x: -kernel(x) @ inner


def experiment():
    exp_kr_prox_grad = KR_PROX_GRAD(
        j=j, p=p, beta=beta, domain=discretization_domain, L=L
    )
    exp_frank_wolfe = FRANK_WOLFE(
        K_transpose=kernel(discretization_domain), beta=beta, target=target
    )

    # KR Prox Grad
    logging.info(f"Computing KR Prox Grad solution")
    u_kr, objective_values_kr, times_kr, supports_kr = exp_kr_prox_grad.solve(
        max_iter=250, max_time=1000, log_results=True
    )
    # logging.info(u_kr.coefficients)
    # for sup, coef in zip(u_kr.support, u_kr.coefficients):
    #     if np.min(np.linalg.norm(true_sources - sup, axis=1))>0.1:
    #         logging.info(f"point: {sup}, coef: {coef}")
    # p_u = p(u_kr)
    # P = lambda x: np.abs(p_u(x))
    # B, D = np.meshgrid(
    #             *(np.linspace(bound[0], bound[1], discretization_resolution + 2)[1:-1] for bound in Omega)
    #         )
    # vals = np.array(
    #     [P(np.array([x_1, x_2])) for x_1, x_2 in zip(B.flatten(), D.flatten())]
    # ).reshape((100, 100))
    # plt.contourf(B, D, vals, levels=100)
    # plt.colorbar()
    # for i, x in enumerate(true_sources):
    #     if i:
    #         plt.plot([x[0]], [x[1]], "P", c="r", markersize=10)
    #     else:
    #         plt.plot([x[0]], [x[1]], "P", c="r", markersize=10, label="True sources")
    # for i, x in enumerate(u_kr.support):
    #     if i:
    #         plt.plot([x[0]], [x[1]], "o", c="b")
    #     else:
    #         plt.plot([x[0]], [x[1]], "o", c="b", label="Predicted support")
    # plt.legend()
    # plt.show()

    # PDAP
    logging.info(f"Computing PDAP solution")
    u_pdap, objective_values_pdap, times_pdap, supports_pdap = (
        exp_frank_wolfe.solve_exact(tol=1e-12, log_results=False)
    )

    optimum = objective_values_pdap[-1]

    residuals_pdap = np.array(objective_values_pdap) - optimum
    residuals_kr_prox_grad = np.array(objective_values_kr) - optimum

    logging.getLogger().setLevel(logging.WARNING)  # Supress logging

    # Plot residuals vs time
    fig, ax = plt.subplots(figsize=(7, 5))
    names = ["PDAP", "KR Prox Grad"]
    styles = ["-", ":"]
    colors = ["red", "blue"]
    for domain, array, name, style, color in zip(
        [times_pdap, times_kr],
        [
            residuals_pdap,
            residuals_kr_prox_grad,
        ],
        names,
        styles,
        colors,
    ):
        ax.semilogy(
            domain,
            array,
            linestyle=style,
            label=name,
            c=color,
        )
    plt.ylabel("Objective residual")
    plt.xlabel("Time (s)")
    plt.ylim(1e-10, 1e2)
    # plt.xlim(0, 1.12)
    ax.legend()
    plt.savefig(results_dir / "res_time.png", bbox_inches="tight")
    plt.close()

    # Plot residuals vs iterations
    fig, ax = plt.subplots(figsize=(7, 5))
    names = ["PDAP", "KR Prox Grad"]
    styles = ["-", ":"]
    colors = ["red", "blue"]
    for array, name, style, color in zip(
        [
            residuals_pdap,
            residuals_kr_prox_grad,
        ],
        names,
        styles,
        colors,
    ):
        ax.semilogy(
            np.arange(len(array)),
            array,
            linestyle=style,
            label=name,
            c=color,
        )
    plt.ylabel("Objective residual")
    plt.xlabel("Iterations")
    plt.ylim(1e-10, 1e2)
    # plt.xlim(0, 100)
    ax.legend()
    plt.savefig(results_dir / "res_iter.png", bbox_inches="tight")
    plt.close()

    # Plot supports
    fig, ax = plt.subplots(figsize=(7, 5))
    names = ["PDAP", "KR Prox Grad"]
    styles = ["-", ":"]
    colors = ["red", "blue"]
    for array, name, style, color in zip(
        [
            supports_pdap,
            supports_kr,
        ],
        names,
        styles,
        colors,
    ):
        ax.plot(np.arange(len(array)), array, linestyle=style, label=name, c=color)
    plt.ylabel("Support points")
    plt.xlabel("Iterations")
    ax.legend()
    plt.savefig(results_dir / "supports.png", bbox_inches="tight")
    plt.close()

    logging.getLogger().setLevel(logging.INFO)  # Reinstate logging


if __name__ == "__main__":
    experiment()
