import numpy as np
import sys
import logging
import pickle
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
from l2_prox_grad import L2_PROX_GRAD
from frank_wolfe import FRANK_WOLFE
from pdap import PDAP

results_dir = Path("results/source_identification")

results_dir.mkdir(parents=True, exist_ok=True)

# Generate data and define functions

Omega = np.array([[0, 1], [0, 1]])
discretization_resolution = 100
beta = 1e-1
observation_resolution = 4
std_factor = 0.05
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

K_transpose = kernel(discretization_domain)

kernel_norm_0 = np.max(np.linalg.norm(K_transpose, axis=1))
kernel_norm_1 = np.max(np.linalg.norm(grad_kernel(discretization_domain), axis=(1, 2)))
L = max(kernel_norm_0, kernel_norm_1) ** 2


def p(u):
    Ku = u.duality_pairing(kernel)
    inner = Ku - target
    return lambda x: -kernel(x) @ inner


def experiment():
    exp_pdap = PDAP(K_transpose=K_transpose, beta=beta, target=target)
    exp_l2_prox_grad = L2_PROX_GRAD(
        K_matrix=K_transpose.T, target=target, beta=beta, L=L
    )
    exp_frank_wolfe = FRANK_WOLFE(
        target=target, K_transpose=K_transpose, beta=beta, L=L
    )
    exp_kr_prox_grad = KR_PROX_GRAD(
        j=j, p=p, beta=beta, domain=discretization_domain, L=L, wasserstein_weight=0.5
    )

    # PDAP
    logging.info(f"Computing PDAP solution")
    u_pdap, objective_values_pdap, times_pdap, supports_pdap = exp_pdap.solve(
        tol=1e-12, log_results=False
    )
    optimum = objective_values_pdap[-1]

    # L2 Prox Grad
    logging.info(f"Computing L2 Prox Grad solution")
    u_l2, objective_values_l2, times_l2, supports_l2 = exp_l2_prox_grad.solve(
        max_iter=1e7, max_time=60, log_results=False, optimum=optimum
    )

    # Frank-Wolfe
    logging.info(f"Computing Frank-Wolfe solution")
    u_fw, objective_values_fw, times_fw, supports_fw = exp_frank_wolfe.solve(
        max_iter=1e7, max_time=60, log_results=False, optimum=optimum
    )

    warm_start = False
    if warm_start:
        with open(f"{results_dir}/6400iter.pkl", "rb") as file:
            u_0 = pickle.load(file)
    else:
        u_0 = Measure()

    # KR Prox Grad
    logging.info(f"Computing KR Prox Grad solution")
    u_kr, objective_values_kr, times_kr, supports_kr = exp_kr_prox_grad.solve(
        max_iter=1e7, max_time=60, log_results=False, mu_0=u_0, optimum=optimum
    )

    with open(f"{results_dir}/6400iter.pkl", "wb") as file:
        pickle.dump(u_kr, file)

    residuals_pdap = np.array(objective_values_pdap) - optimum
    residuals_l2 = np.array(objective_values_l2) - optimum
    residuals_fw = np.array(objective_values_fw) - optimum
    residuals_kr = np.array(objective_values_kr) - optimum

    logging.getLogger().setLevel(logging.WARNING)  # Supress logging

    # Plot residuals vs time
    fig, ax = plt.subplots(figsize=(7, 5))
    names = ["PDAP", "L2 Prox Grad", "Frank-Wolfe", "KR Prox Grad"]
    styles = ["-", "-.", "--", ":"]
    colors = ["red", "green", "orange", "blue"]
    for domain, array, name, style, color in zip(
        [times_pdap, times_l2, times_fw, times_kr],
        [
            residuals_pdap,
            residuals_l2,
            residuals_fw,
            residuals_kr,
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
    names = ["PDAP", "L2 Prox Grad", "Frank-Wolfe", "KR Prox Grad"]
    styles = ["-", "-.", "--", ":"]
    colors = ["red", "green", "orange", "blue"]
    for array, name, style, color in zip(
        [
            residuals_pdap,
            residuals_l2,
            residuals_fw,
            residuals_kr,
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
    names = ["PDAP", "L2 Prox Grad", "Frank-Wolfe", "KR Prox Grad"]
    styles = ["-", "-.", "--", ":"]
    colors = ["red", "green", "orange", "blue"]
    for array, name, style, color in zip(
        [
            supports_pdap,
            supports_l2,
            supports_fw,
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
