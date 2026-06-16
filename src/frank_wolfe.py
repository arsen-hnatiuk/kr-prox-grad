import numpy as np
import logging
import time
from lib.default_values import *
from lib.measure import Measure

logging.basicConfig(
    level=logging.DEBUG,
)


class FRANK_WOLFE:
    def __init__(
        self,
        target: np.ndarray,
        K_matrix: np.ndarray,
        beta: float,
        L: float = 1,
    ) -> None:
        self.K_matrix = K_matrix
        self.domain_size = self.K_matrix.shape[1]
        self.beta = beta
        self.target = target
        self.residuum = lambda mu: self.K_matrix @ mu - self.target
        self.gradient = lambda mu: self.K_matrix.T @ self.residuum(mu) + self.beta
        self.f = lambda mu: 0.5 * np.sum(np.square(self.residuum(mu)))
        self.g = lambda mu: beta * np.sum(mu)
        self.j = lambda mu: self.f(mu) + self.g(mu)
        self.L = L
        self.L_max = np.inf
        self.L_min = 0
        self.L_reduce_factor = 0.9
        self.L_increase_factor = 2
        self.M = 1
        self.tol = 1e-8

    def frank_wolfe_step(self, mu: np.ndarray) -> tuple:
        grad_mu = self.gradient(mu)

        # New support candidate
        s_index = np.argmin(grad_mu)
        min_val = grad_mu[s_index]
        if min_val >= 0:
            s = np.zeros(self.domain_size)
        else:
            s = self.M * np.eye(1, self.domain_size, s_index)[0]
        max_step = 1
        update_direction = s - mu
        grad_update_direction = -grad_mu @ update_direction
        norm_update = np.sum(np.square(update_direction))

        # Line search
        self.L = max(self.L * self.L_reduce_factor, self.L_min)
        descent_condition = False
        while not descent_condition:
            step_size = min(max_step, grad_update_direction / (self.L * norm_update))
            mu_plus = mu + step_size * update_direction

            # Check descent
            diff = self.j(mu_plus) - self.j(mu)
            descent_rhs = (
                -step_size * grad_update_direction
                + 0.5 * step_size**2 * self.L * norm_update
            )
            if diff <= descent_rhs and diff <= 0:
                descent_condition = True
            if not descent_condition:
                self.L = min(self.L * self.L_increase_factor, self.L_max)

        return mu_plus

    def solve(
        self,
        max_time: float = 60.0,
        max_iter: int = 250,
        log_results: bool = True,
        optimum: float = np.inf,
        exit_tol: float = 1e-10,
        mu_0: np.ndarray = np.zeros(0),
    ) -> tuple:
        if not len(mu_0):
            mu = np.zeros(self.domain_size)
        else:
            mu = mu_0
        k = 1
        initial_time = time.perf_counter()
        objectives = [self.j(mu)]
        times = [time.perf_counter() - initial_time]
        supports = [np.sum(mu != 0)]
        while time.perf_counter() - initial_time < max_time and k <= max_iter:
            self.M = self.j(mu) / self.beta

            mu_plus = self.frank_wolfe_step(mu)

            mu = mu_plus.copy()
            if np.any(mu <= -self.tol):
                logging.warning("Negative coefficients")

            objectives.append(self.j(mu))
            times.append(time.perf_counter() - initial_time)
            supports.append(np.sum(mu != 0))

            if objectives[-1] > objectives[-2] * (1 + self.tol):
                logging.warning("Ascent observed")
            if log_results:
                logging.info(
                    f"{k}: L:{self.L:.3E}, support {supports[-1]}, objective: {objectives[-1]:.12E}"
                )
            # Check optimality
            if abs(objectives[-1] - optimum) < exit_tol:
                break
            k += 1
        logging.info(
            f"Frank-Wolfe exited after {k} iterations and {times[-1]:.3f}s with final sparsity of {supports[-1]} and objective {objectives[-1]:.12E}"
        )
        return mu, objectives, times, supports
