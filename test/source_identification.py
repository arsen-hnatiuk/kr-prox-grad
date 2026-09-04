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


def generate_data(discretization_resolution: int) -> tuple:

    Omega = np.array([[0, 1], [0, 1]])
    beta = 1e-1
    observation_resolution = 4
    std_factor = 0.05
    true_sources = np.array([[0.28, 0.71], [0.51, 0.27], [0.71, 0.53]])
    true_weights = np.array([1, 0.7, 0.8])

    def get_grid(size: int) -> np.ndarray:
        grid = (
            np.array(
                np.meshgrid(
                    *(
                        np.linspace(bound[0], bound[1], size + 2)[1:-1]
                        for bound in Omega
                    )
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
            gradient = diff * exponentiated.reshape(
                -1, 1
            )  # shape=(len(x),Omega.shape[0])
            gradients.append(gradient)
        result = (
            np.transpose(np.array(gradients), axes=(1, 0, 2)) / outer_factor
        )  # The Jacobian of kappa, shape=(len(x), len(observations), Omega.shape[0])
        return result

    u_hat = Measure(support=true_sources, coefficients=true_weights)
    target = u_hat.duality_pairing(kernel)
    K_transpose = kernel(discretization_domain)

    def forward_operator(u: Measure):
        return u.duality_pairing(K_transpose)

    g = lambda u: beta * np.linalg.norm(u, ord=1)
    f = lambda y: 0.5 * np.sum((y - target) ** 2)

    def j(u: Measure, Ku: list):
        return f(Ku) + g(u.coefficients)

    kernel_norm_0 = np.max(np.linalg.norm(K_transpose, axis=1))
    kernel_norm_1 = np.max(
        np.linalg.norm(grad_kernel(discretization_domain), axis=(1, 2))
    )
    L = max(kernel_norm_0, kernel_norm_1) ** 2

    def p(Ku):
        inner = Ku - target
        return -K_transpose @ inner

    return K_transpose, beta, target, L, j, p, discretization_domain, forward_operator


def experiment():
    logging.info("Running the algorithms on different meshes")
    mesh_sizes = [50, 100, 250]
    for discretization_resolution in mesh_sizes:
        K_transpose, beta, target, L, j, p, discretization_domain, forward_operator = (
            generate_data(discretization_resolution)
        )
        exp_pdap = PDAP(K_transpose=K_transpose, beta=beta, target=target)
        exp_l2_prox_grad = L2_PROX_GRAD(
            K_matrix=K_transpose.T, target=target, beta=beta, L=L
        )
        exp_frank_wolfe = FRANK_WOLFE(target=target, K_matrix=K_transpose.T, beta=beta)
        exp_kr_prox_grad = KR_PROX_GRAD(
            j=j,
            p=p,
            beta=beta,
            forward_operator=forward_operator,
            domain=discretization_domain,
            L=L,
            wasserstein_weight=1,
        )

        # PDAP
        logging.info(f"Computing PDAP solution")
        u_pdap, objective_values_pdap, times_pdap, supports_pdap = exp_pdap.solve(
            tol=1e-12, log_results=False
        )
        optimum = objective_values_pdap[-1]
        if discretization_resolution == 100:
            optimum_100 = optimum

        # L2 Prox Grad
        logging.info(f"Computing L2 Prox Grad solution")
        u_l2, objective_values_l2, times_l2, supports_l2 = exp_l2_prox_grad.solve(
            max_iter=1e7, max_time=600, log_results=False, optimum=optimum
        )
        l2_residual = objective_values_l2[-1] - optimum
        l2_time = times_l2[-1]
        if discretization_resolution == 100:
            times_l2_plot = times_l2
            residuals_l2 = np.array(objective_values_l2) - optimum
            supports_l2_plot = supports_l2

        # Frank-Wolfe
        logging.info(f"Computing Frank-Wolfe solution")
        u_fw, objective_values_fw, times_fw, supports_fw = exp_frank_wolfe.solve(
            max_iter=1e7, max_time=600, log_results=False, optimum=optimum
        )
        fw_residual = objective_values_fw[-1] - optimum
        fw_time = times_fw[-1]
        if discretization_resolution == 100:
            times_fw_plot = times_fw
            residuals_fw = np.array(objective_values_fw) - optimum
            supports_fw_plot = supports_fw

        # warm_start = False
        # if warm_start:
        #     with open(f"{results_dir}/6400iter.pkl", "rb") as file:
        #         u_0 = pickle.load(file)
        # else:
        #     u_0 = Measure()

        # KR Prox Grad
        logging.info(f"Computing KR Prox Grad solution")
        u_kr, objective_values_kr, times_kr, supports_kr = exp_kr_prox_grad.solve(
            max_iter=1e7,
            max_time=600,
            log_results=False,
            mu_0=Measure(),
            optimum=optimum,
        )
        kr_residual = objective_values_kr[-1] - optimum
        kr_time = times_kr[-1]
        if discretization_resolution == 100:
            times_kr_plot = times_kr
            residuals_kr = np.array(objective_values_kr) - optimum
            supports_kr_plot = supports_kr

        logging.info(
            f"Mesh size: {discretization_resolution}: L2 time {l2_time:.3E}, L2 residual {l2_residual}; FW time {fw_time:.3E}, FW residual {fw_residual}; KR time {kr_time:.3E}, KR residual {kr_residual}"
        )
        logging.info("-" * 75)

        # with open(f"{results_dir}/6400iter.pkl", "wb") as file:
        #     pickle.dump(u_kr, file)

    logging.info("Running KR prox grad using different wasserstein weights")
    K_transpose, beta, target, L, j, p, discretization_domain, forward_operator = (
        generate_data(discretization_resolution=100)
    )
    wasserstein_weights = [0.25, 0.75, 2.0]
    all_times_kr = []
    all_residuals_kr = []
    all_supports_kr = []
    for wasserstein_weight in wasserstein_weights:
        exp_kr_prox_grad = KR_PROX_GRAD(
            j=j,
            p=p,
            beta=beta,
            domain=discretization_domain,
            L=L,
            wasserstein_weight=wasserstein_weight,
        )
        logging.info(
            f"Computing KR Prox Grad solution with weight {wasserstein_weight}"
        )
        u_kr, objective_values_kr, times_kr, supports_kr = exp_kr_prox_grad.solve(
            max_iter=1e7,
            max_time=600,
            log_results=False,
            mu_0=Measure(),
            optimum=optimum_100,
        )
        all_times_kr.append(times_kr)
        all_residuals_kr.append(np.array(objective_values_kr) - optimum_100)
        all_supports_kr.append(supports_kr)
    all_times_kr = all_times_kr[:2] + [times_kr_plot] + all_times_kr[-1:]
    all_residuals_kr = all_residuals_kr[:2] + [residuals_kr] + all_residuals_kr[-1:]
    all_supports_kr = all_supports_kr[:2] + [supports_kr_plot] + all_supports_kr[-1:]
    wasserstein_weights = [0.25, 0.75, 1.0, 2.0]

    logging.getLogger().setLevel(logging.WARNING)  # Supress logging

    # Plot residuals vs time
    fig, ax = plt.subplots(figsize=(7, 5))
    names = ["L2 Prox Grad", "Frank-Wolfe"]
    styles = ["-.", "--"]
    colors = ["green", "orange"]
    for domain, array, name, style, color in zip(
        [times_l2_plot, times_fw_plot],
        [
            residuals_l2,
            residuals_fw,
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
    plt.savefig(results_dir / "res_time_other.png", bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 5))
    names = [
        rf"KR Prox Grad, $\vartheta={float(wasserstein_weight)}$"
        for wasserstein_weight in wasserstein_weights
    ]
    styles = ["-", "-.", "--", ":"]
    colors = ["lightblue", "dodgerblue", "blue", "black"]
    for domain, array, name, style, color in zip(
        all_times_kr,
        all_residuals_kr,
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
    plt.savefig(results_dir / "res_time_kr.png", bbox_inches="tight")
    plt.close()

    # Plot residuals vs iterations
    fig, ax = plt.subplots(figsize=(7, 5))
    names = ["L2 Prox Grad", "Frank-Wolfe"]
    styles = ["-.", "--"]
    colors = ["green", "orange"]
    for array, name, style, color in zip(
        [
            residuals_l2,
            residuals_fw,
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
    plt.savefig(results_dir / "res_iter_other.png", bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 5))
    names = [
        rf"KR Prox Grad, $\vartheta={float(wasserstein_weight)}$"
        for wasserstein_weight in wasserstein_weights
    ]
    styles = ["-", "-.", "--", ":"]
    colors = ["lightblue", "dodgerblue", "blue", "black"]
    for array, name, style, color in zip(
        all_residuals_kr,
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
    plt.savefig(results_dir / "res_iter_kr.png", bbox_inches="tight")
    plt.close()

    # Plot supports
    fig, ax = plt.subplots(figsize=(7, 5))
    names = ["L2 Prox Grad", "Frank-Wolfe"]
    styles = ["-.", "--"]
    colors = ["green", "orange"]
    for array, name, style, color in zip(
        [
            supports_l2_plot,
            supports_fw_plot,
        ],
        names,
        styles,
        colors,
    ):
        ax.loglog(np.arange(len(array)), array, linestyle=style, label=name, c=color)
    plt.ylabel("Support points")
    plt.xlabel("Iterations")
    ax.legend()
    plt.savefig(results_dir / "supports_other.png", bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 5))
    names = [
        rf"KR Prox Grad, $\vartheta={float(wasserstein_weight)}$"
        for wasserstein_weight in wasserstein_weights
    ]
    styles = ["-", "-.", "--", ":"]
    colors = ["lightblue", "dodgerblue", "blue", "black"]
    for array, name, style, color in zip(
        all_supports_kr,
        names,
        styles,
        colors,
    ):
        ax.semilogx(np.arange(len(array)), array, linestyle=style, label=name, c=color)
    plt.ylabel("Support points")
    plt.xlabel("Iterations")
    ax.legend()
    plt.savefig(results_dir / "supports_kr.png", bbox_inches="tight")
    plt.close()

    logging.getLogger().setLevel(logging.INFO)  # Reinstate logging

    # # Plot splitting and transport
    # K_transpose, beta, target, L, j, p, discretization_domain = generate_data(
    #     discretization_resolution=100
    # )
    # exp_kr_prox_grad = KR_PROX_GRAD(
    #     j=j,
    #     p=p,
    #     beta=beta,
    #     domain=discretization_domain,
    #     L=L,
    #     wasserstein_weight=0.5,
    #     transport_plot=True,
    # )
    # mu, mu_plus, plotting_dict = exp_kr_prox_grad.solve(
    #     max_iter=1e7,
    #     max_time=60,
    #     log_results=False,
    #     mu_0=Measure(),
    #     optimum=0,
    # )
    # j = 0
    # selected_target = []
    # for j_, inner_dict in plotting_dict.items():
    #     transported_to = inner_dict["transported_to"]
    #     if len(transported_to) > len(selected_target):
    #         j = j_
    #         selected_target = transported_to
    # initial_point = mu.support[j_]
    # plot_width = 0.1
    # plot_bounds = [
    #     [initial_point[0] - plot_width, initial_point[0] + plot_width],
    #     [initial_point[1] - plot_width, initial_point[1] + plot_width],
    # ]
    # overlap_domain = []
    # for point in discretization_domain:
    #     if (
    #         point[0] > plot_bounds[0][0] + 0.0011
    #         and point[0] < plot_bounds[0][1] - 0.0011
    #         and point[1] > plot_bounds[1][0] + 0.0011
    #         and point[1] < plot_bounds[1][1] - 0.0011
    #     ):
    #         overlap_domain.append(point)
    # p_mu = p(mu)
    # B, D = np.meshgrid(
    #     *(np.linspace(plot_bounds[_][0], plot_bounds[_][1], 100) for _ in range(2))
    # )
    # vals = np.array(
    #     [p_mu(np.array([x_1, x_2])) for x_1, x_2 in zip(B.flatten(), D.flatten())]
    # ).reshape((100, 100))

    # logging.getLogger().setLevel(logging.WARNING)  # Supress logging

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    # cs1 = ax1.contourf(B, D, vals, levels=100)
    # for i, x in enumerate(overlap_domain):
    #     if i:
    #         ax1.scatter(x[0], x[1], marker="o", c="silver", s=5)
    #     else:
    #         ax1.scatter(x[0], x[1], marker="o", c="silver", s=5, label="Mesh points")
    # for i, x in enumerate(mu.support):
    #     if i:
    #         ax1.scatter(x[0], x[1], marker="o", c="tomato", s=30)
    #     else:
    #         ax1.scatter(
    #             x[0],
    #             x[1],
    #             marker="o",
    #             c="tomato",
    #             s=30,
    #             label=r"Support points of $\mu$",
    #         )
    # ax1.scatter(
    #     initial_point[0],
    #     initial_point[1],
    #     marker="o",
    #     c="crimson",
    #     s=50,
    #     label="Transport origin",
    #     edgecolors="black",
    # )
    # ax1.legend()
    # ax1.set_xlim(plot_bounds[0][0], plot_bounds[0][1])
    # ax1.set_ylim(plot_bounds[1][0], plot_bounds[1][1])

    # cs2 = ax2.contourf(B, D, vals, levels=100)
    # for i, x in enumerate(overlap_domain):
    #     if i:
    #         ax2.scatter(x[0], x[1], marker="o", c="silver", s=5)
    #     else:
    #         ax2.scatter(x[0], x[1], marker="o", c="silver", s=5, label="Mesh points")
    # for i, x in enumerate(mu_plus.support):
    #     if i:
    #         ax2.scatter(x[0], x[1], marker="o", c="tomato", s=30)
    #     else:
    #         ax2.scatter(
    #             x[0],
    #             x[1],
    #             marker="o",
    #             c="tomato",
    #             s=30,
    #             label=r"Support points of $\mu_+$",
    #         )
    # for i, x in enumerate(selected_target):
    #     if i:
    #         ax2.scatter(x[0], x[1], marker="o", c="crimson", s=50, edgecolors="black")
    #     else:
    #         ax2.scatter(
    #             x[0],
    #             x[1],
    #             marker="o",
    #             c="crimson",
    #             s=50,
    #             label="Tansport destination",
    #             edgecolors="black",
    #         )
    # ax2.set_xlim(plot_bounds[0][0], plot_bounds[0][1])
    # ax2.set_ylim(plot_bounds[1][0], plot_bounds[1][1])
    # ax2.legend()

    # # cbar = fig.colorbar(cs1, ax=[ax1, ax2])
    # plt.tight_layout()
    # fig.savefig(results_dir / "transport.png", bbox_inches="tight")
    plt.close()

    logging.getLogger().setLevel(logging.INFO)  # Reinstate logging


if __name__ == "__main__":
    experiment()
