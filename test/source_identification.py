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
from pdap import PDAP

results_dir = Path("results/source_identification")

results_dir.mkdir(parents=True, exist_ok=True)

# Generate data and define functions

Omega = np.array([[0, 1], [0, 1]])
alpha = 1e-1
observation_resolution = 4
std_factor = 0.1
true_sources = np.array([[0.28, 0.71], [0.51, 0.27], [0.71, 0.53]])
true_weights = np.array([1, -0.7, 0.8])


def get_grid(size: int) -> np.ndarray:
    grid = (
        np.array(
            np.meshgrid(*(np.linspace(bound[0], bound[1], size + 2) for bound in Omega))
        )
        .reshape(len(Omega), -1)
        .T
    )
    return grid


observations = get_grid(observation_resolution)


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


def hess_kernel(x):
    # Input is 2D array of shape (number of points, Omega dimension)
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    hessians = []
    outer_factors = [
        -std_factor * np.sqrt(std_factor * np.pi) ** Omega.shape[0] / 2,
        std_factor**2 * np.sqrt(std_factor * np.pi) ** Omega.shape[0] / 4,
    ]
    for point in observations:
        diff = point - x  # (len(x), Omega.shape[0])
        norms = -np.square(np.linalg.norm(diff, axis=1)) / std_factor  # (len(x),)
        exponentiated_normed_1 = np.exp(norms) / outer_factors[0]  # (len(x),)
        exponentiated_normed_2 = np.exp(norms) / outer_factors[1]  # (len(x),)
        first_part = np.repeat(
            np.eye(Omega.shape[0])[np.newaxis, :], len(x), axis=0
        ) * exponentiated_normed_1.reshape(
            -1, 1, 1
        )  # shape=(len(x),Omega.shape[0],Omega.shape[0])
        second_part = np.einsum(
            "ij,ik->ijk", diff, diff
        ) * exponentiated_normed_2.reshape(
            -1, 1, 1
        )  # shape=(len(x),Omega.shape[0],Omega.shape[0])
        hessians.append(first_part + second_part)
    result = np.transpose(
        np.array(hessians), axes=(1, 0, 2, 3)
    )  # The derivative of the Jacobian of kappa, shape=(len(x), len(observations), Omega.shape[0], Omega.shape[0])
    return result


u_hat = Measure(support=true_sources, coefficients=true_weights)
target = u_hat.duality_pairing(kernel)

g = lambda u: alpha * np.linalg.norm(u, ord=1)
f = lambda y: 0.5 * np.linalg.norm(y - target) ** 2
grad_f = lambda x: x - target
hess_f = lambda x: np.identity(len(x))
j = lambda u: f(u.duality_pairing(kernel)) + g(u.coefficients)


def p(u):
    Ku = u.duality_pairing(kernel)
    inner = Ku - target
    return lambda x: -kernel(x) @ inner


def grad_p(u):
    inner = target - u.duality_pairing(kernel)
    return lambda x: np.tensordot(grad_kernel(x), inner, axes=([1, 0]))


def hess_p(u):
    inner = target - u.duality_pairing(kernel)
    return lambda x: np.tensordot(hess_kernel(x), inner, axes=([1, 0]))


def j_N(coefs, positions):
    K_matrix = kernel(positions)
    grad_F = (K_matrix.T @ coefs).flatten() - target
    return 0.5 * np.linalg.norm(grad_F) ** 2 + alpha * np.linalg.norm(coefs, ord=1)


def grad_j_N(coefs, positions):
    K_matrix = kernel(positions)
    grad_F = (K_matrix.T @ coefs).flatten() - target
    nabla_x = coefs.reshape(-1, 1) * np.tensordot(
        grad_kernel(positions), grad_F, axes=([1, 0])
    )
    nabla_u = np.dot(K_matrix, grad_F) + alpha * np.sign(coefs)
    return np.append(nabla_x.flatten(), nabla_u, axis=0).flatten()


def hess_f_N(coefs, positions):
    kappa_values = kernel(positions)
    grad_kappa_values = grad_kernel(positions)
    hess_kappa_values = hess_kernel(positions)
    matrix_dimension = len(positions) * Omega.shape[0] + len(coefs)
    hesse_matrix = np.zeros((matrix_dimension, matrix_dimension))
    step = Omega.shape[0]
    coefs_delay = step * len(positions)
    inner = (kappa_values.T @ coefs).flatten() - target
    for i in range(len(positions)):
        # nabla_{x_i,x_j}
        for j in range(len(positions)):
            if j < i:
                continue
            block = (
                coefs[i]
                * coefs[j]
                * np.matmul(grad_kappa_values[i].T, grad_kappa_values[j])
            )
            if i == j:
                block += coefs[i] * np.tensordot(
                    hess_kappa_values[i], inner, axes=([0, 0])
                )
            hesse_matrix[i * step : (i + 1) * step, j * step : (j + 1) * step] = block
            hesse_matrix[j * step : (j + 1) * step, i * step : (i + 1) * step] = block.T
        # nabla_{x_i,u_j}
        for j in range(len(coefs)):
            block = coefs[i] * np.matmul(grad_kappa_values[i].T, kappa_values[j])
            if i == j:
                block += np.matmul(grad_kappa_values[i].T, inner)
            hesse_matrix[i * step : (i + 1) * step, coefs_delay + j] = block
            hesse_matrix[coefs_delay + j, i * step : (i + 1) * step] = block.T
    for i in range(len(coefs)):
        # nabla_{u_i,u_j}
        for j in range(len(coefs)):
            if j < i:
                continue
            block = np.dot(kappa_values[i], kappa_values[j])
            hesse_matrix[coefs_delay + i, coefs_delay + j] = block
            hesse_matrix[coefs_delay + j, coefs_delay + i] = block
    return hesse_matrix


def experiment():
    exp_kr_prox_grad = KR_PROX_GRAD(
        target=target,
        kernel=kernel,
        g=g,
        f=f,
        grad_f=grad_f,
        hess_f=hess_f,
        hess_f_N=hess_f_N,
        j=j,
        j_N=j_N,
        p=p,
        grad_p=grad_p,
        hess_p=hess_p,
        grad_j_N=grad_j_N,
        alpha=alpha,
        Omega=Omega,
        global_search_resolution=10,
        sample_size=500,
        R=0.1,
    )
    finite_grid = get_grid(100)
    K_transpose = kernel(finite_grid)
    exp_pdap = PDAP(K_transpose=K_transpose, alpha=alpha, target=target)

    # PDAP
    logging.info(f"Computing PDAP solution")
    u_pdap, objective_values_pdap, times_pdap, supports_pdap = exp_pdap.solve_exact(
        tol=1e-10, do_logging=True
    )

    # KR Prox Grad
    logging.info(f"Computing KR Prox Grad solution")
    (
        u_kr_prox_grad,
        times_kr_prox_grad,
        supports_kr_prox_grad,
        objective_values_kr_prox_grad,
    ) = exp_kr_prox_grad.solve(
        tol=1e-12, sampling="deterministic", mode="lazy", log_results=False
    )

    optimum = (
        min(
            [
                objective_values_pdap[-1],
                objective_values_kr_prox_grad[-1],
            ]
        )
        - 1e-13
    )

    residuals_pdap = np.array(objective_values_pdap) - optimum
    residuals_kr_prox_grad = np.array(objective_values_kr_prox_grad) - optimum

    logging.getLogger().setLevel(logging.WARNING)  # Supress logging

    # Plot residuals vs time
    fig, ax = plt.subplots(figsize=(7, 5))
    names = ["PDAP", "KR Prox Grad"]
    styles = ["-", ":"]
    colors = ["red", "blue"]
    for domain, array, name, style, color in zip(
        [times_pdap, times_kr_prox_grad][
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
    plt.ylim(1e-10, 1e1)
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
    plt.ylim(1e-10, 1e1)
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
            supports_kr_prox_grad,
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
