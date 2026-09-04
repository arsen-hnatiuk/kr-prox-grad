"""
1D resolution of the initial state of a non-linear Fisher equation
"""

import numpy as np
import sys
import logging
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Callable
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem

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

results_dir = Path("results/non_linear_source_identification")

results_dir.mkdir(parents=True, exist_ok=True)

Omega = np.array([[0, 1]])
beta = 1e-1
time_resolution = 100
convolution_variance = 0.0005
true_sources = np.array([0.28, 0.51, 0.57])
true_weights = np.array([1, 0.7, 0.8])
L = 100
r = 1  # Growth parameter
D = 0.01  # Diffusion parameter
optimum = 0


def solve_adjoint_pde(
    initial_condition: np.ndarray,
    domain: mesh.Mesh,
    forward_solutions: list,
    log_results: bool = False,
) -> np.ndarray:
    # Solved in direct time by reversing the equation
    dt = 1 / time_resolution

    # Define FE
    V = fem.functionspace(domain, ("Lagrange", 1))
    fdim = domain.topology.dim - 1

    # Initial condition
    w_0 = fem.Function(V)
    w_0.name = "w_0"
    w_0.x.array[:] = initial_condition

    # Boundary condition
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
    )
    bc = fem.dirichletbc(
        PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V
    )

    # Variational form problem
    w = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    w_new = fem.Function(V)
    forward_solution = fem.Function(V)
    bilinear = (
        (w / dt) * v
        + D * ufl.dot(ufl.grad(w), ufl.grad(v))
        - r * (1 - 2 * forward_solution) * w * v
    ) * ufl.dx
    linear = (w_0 * v / dt) * ufl.dx
    problem = LinearProblem(
        bilinear,
        linear,
        bcs=[bc],
        u=w_new,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="adjoint_",
    )

    if log_results:
        logging.getLogger().setLevel(logging.WARNING)  # Supress logging
        plt.plot(domain.geometry.x[:, 0].flatten(), w_0.x.array[:])
        plt.show()
        logging.getLogger().setLevel(logging.INFO)  # Reinstate logging

    for time_step in range(time_resolution):
        forward_solution.x.array[:] = forward_solutions[time_resolution - time_step - 1]
        problem.solve()
        w_0.x.array[:] = w_new.x.array
        if log_results:
            if not (time_step + 1) % 10:
                logging.getLogger().setLevel(logging.WARNING)  # Supress logging
                plt.plot(domain.geometry.x[:, 0].flatten(), w_0.x.array[:])
                plt.show()
                logging.getLogger().setLevel(logging.INFO)  # Reinstate logging

    return w_new.x.array.copy()


def solve_nonlinear_pde(
    initial_condition: np.ndarray, domain: mesh.Mesh, log_results: bool = False
) -> list:
    dt = 1 / time_resolution
    solutions = []

    # Define FE
    V = fem.functionspace(domain, ("Lagrange", 1))
    fdim = domain.topology.dim - 1

    # Initial condition
    w_0 = fem.Function(V)
    w_0.name = "w_0"
    w_0.x.array[:] = initial_condition

    # Boundary condition
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
    )
    bc = fem.dirichletbc(
        PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V
    )

    # Weak form problem
    w = fem.Function(V)
    v = ufl.TestFunction(V)
    F = (
        (w - w_0) / dt * v + D * ufl.dot(ufl.grad(w), ufl.grad(v)) - r * w * (1 - w) * v
    ) * ufl.dx
    problem = NonlinearProblem(
        F,
        w,
        bcs=[bc],
        petsc_options={
            "snes_type": "newtonls",
            "snes_rtol": 1e-8,
            "snes_atol": 1e-10,
            "snes_max_it": 50,
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
        petsc_options_prefix="fisher",
    )

    if log_results:
        logging.getLogger().setLevel(logging.WARNING)  # Supress logging
        plt.plot(domain.geometry.x[:, 0].flatten(), w_0.x.array[:])
        plt.show()
        logging.getLogger().setLevel(logging.INFO)  # Reinstate logging

    for time_step in range(time_resolution):
        w.x.array[:] = w_0.x.array  # initiate solution
        problem.solve()
        converged = problem.solver.getConvergedReason()
        num_iter = problem.solver.getIterationNumber()
        assert converged > 0, f"Solver did not converge, got {converged}."
        w_0.x.array[:] = w.x.array
        solutions.append(w.x.array.copy())
        if log_results:
            logging.info(
                f"Solver converged after {num_iter} iterations with converged reason {converged}."
            )
            if not (time_step + 1) % 10:
                logging.getLogger().setLevel(logging.WARNING)  # Supress logging
                plt.plot(domain.geometry.x[:, 0].flatten(), w_0.x.array[:])
                plt.show()
                logging.getLogger().setLevel(logging.INFO)  # Reinstate logging

    return solutions


# Generate data and define functions
def generate_data(discretization_resolution: int) -> tuple:
    domain = mesh.create_interval(
        MPI.COMM_WORLD, discretization_resolution, [Omega[0][0], Omega[0][1]]
    )
    discretization_domain = domain.geometry.x[:, 0].reshape(-1, Omega.shape[0])

    def raw_convolution_kernel(diff: np.ndarray) -> np.ndarray:
        outer = 1 / np.sqrt(2 * convolution_variance * np.pi)
        inner = -(diff**2) / (2 * convolution_variance)
        return outer * np.exp(inner)

    def initial_condition(u: Measure) -> Callable:
        return np.sum(
            [
                coef * raw_convolution_kernel(discretization_domain.flatten() - pos)
                for pos, coef in zip(u.support.flatten(), u.coefficients)
            ],
            axis=0,
        )

    def convolution_kernel(
        space_domain: np.ndarray, observations: np.ndarray
    ) -> np.ndarray:
        differences = space_domain[:, None] - observations.flatten()
        return raw_convolution_kernel(
            differences
        )  # shape(len(discretization_domain), len(observations)))

    convolution_matrix = convolution_kernel(
        discretization_domain.flatten(), discretization_domain.flatten()
    )

    def forward_operator(mu: Measure):
        pde_solutions = solve_nonlinear_pde(initial_condition(mu), domain)
        return pde_solutions

    mu_hat = Measure(support=true_sources, coefficients=true_weights)
    target_functions = solve_nonlinear_pde(initial_condition(mu_hat), domain)
    target = target_functions[-1]

    g = lambda mu: beta * np.linalg.norm(mu, ord=1)
    f = lambda y: 0.5 * np.sum((y - target) ** 2)

    def j(mu: Measure, Kmu: list):
        inner = Kmu[-1]
        return f(inner) + g(mu.coefficients)

    def p(Kmu):
        inner = Kmu[-1] - target
        adjoint = solve_adjoint_pde(
            initial_condition=inner, domain=domain, forward_solutions=Kmu
        )
        return convolution_matrix @ adjoint

    # mu = Measure()
    # # forward_solutions = solve_nonlinear_pde(initial_condition(mu), domain, True)
    # # inner = forward_solutions[-1] - target
    # # adjoint = solve_adjoint_pde(
    # #     initial_condition=inner,
    # #     domain=domain,
    # #     forward_solutions=forward_solutions,
    # #     log_results=True,
    # # )
    # Kmu = forward_operator(mu)
    # p_mu = p(Kmu)
    # logging.getLogger().setLevel(logging.WARNING)  # Supress logging
    # plt.plot(discretization_domain.flatten(), p_mu)
    # plt.show()
    # logging.getLogger().setLevel(logging.INFO)  # Supress logging

    return beta, L, j, p, discretization_domain, forward_operator


def experiment():
    mesh_sizes = [100]
    for discretization_resolution in mesh_sizes:
        beta, L, j, p, discretization_domain, forward_operator = generate_data(
            discretization_resolution
        )

        exp_kr_prox_grad = KR_PROX_GRAD(
            j=j,
            p=p,
            forward_operator=forward_operator,
            beta=beta,
            domain=discretization_domain,
            L=L,
            wasserstein_weight=1,
        )

        # KR Prox Grad
        logging.info(f"Computing KR Prox Grad solution")
        mu_kr, objective_values_kr, times_kr, supports_kr = exp_kr_prox_grad.solve(
            max_iter=1e7,
            max_time=600,
            log_results=True,
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
            f"Mesh size: {discretization_resolution}: KR time {kr_time:.3E}, KR residual {kr_residual}"
        )
        logging.info("-" * 75)

        # with open(f"{results_dir}/6400iter.pkl", "wb") as file:
        #     pickle.dump(u_kr, file)

    logging.getLogger().setLevel(logging.WARNING)  # Supress logging

    # Plot residuals vs time
    fig, ax = plt.subplots(figsize=(5, 5))
    names = [rf"KR Prox Grad, $\vartheta={1}$"]
    styles = ["--"]
    colors = ["blue"]
    for domain, array, name, style, color in zip(
        times_kr,
        residuals_kr,
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
    fig, ax = plt.subplots(figsize=(5, 5))
    names = [rf"KR Prox Grad, $\vartheta={1}$"]
    styles = ["--"]
    colors = ["blue"]
    for array, name, style, color in zip(
        residuals_kr,
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
    fig, ax = plt.subplots(figsize=(5, 5))
    names = [rf"KR Prox Grad, $\vartheta={1}$"]
    styles = ["--"]
    colors = ["blue"]
    for array, name, style, color in zip(
        supports_kr,
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


if __name__ == "__main__":
    experiment()
