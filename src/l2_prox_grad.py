import numpy as np
import logging
import time

logging.basicConfig(
    level=logging.DEBUG,
)


class L2_PROX_GRAD:

    def __init__(
        self,
        K_matrix: np.ndarray,
        target: np.ndarray,
        beta: float,
        L: float,
        tol: float = 1e-8,
    ) -> None:
        self.K_matrix = K_matrix
        self.target = target
        self.residuum = lambda mu: self.K_matrix @ mu - self.target
        self.f = lambda mu: 0.5 * np.sum(np.square(self.residuum(mu)))
        self.beta = beta
        self.g = lambda mu: self.beta * np.linalg.norm(mu, ord=1)
        self.j = lambda mu: self.f(mu) + self.g(mu)
        self.L = L
        self.L_max = np.inf
        self.L_min = 1
        self.L_reduce_factor = 0.9
        self.L_increase_factor = 2
        self.tol = tol

    def prox_step(self, mu: np.ndarray, p_mu: np.ndarray, varphi: np.ndarray) -> tuple:
        mu_plus = np.maximum(mu - varphi / self.L, 0)
        l2_norm = np.linalg.norm(mu_plus)

        # Check descent
        descent_condition = True
        diff = self.j(mu_plus) - self.j(mu)
        descent_rhs = (
            -p_mu @ (mu_plus - mu)
            + self.g(mu_plus)
            - self.g(mu)
            + 0.5 * self.L * l2_norm**2
        )
        if diff > descent_rhs or diff > 0:
            descent_condition = False
        return mu_plus, descent_condition

    def solve(
        self,
        mu_0: np.ndarray = np.zeros(0),
        max_time: float = 60.0,
        max_iter: int = 250,
        log_results: bool = True,
        optimum: float = np.inf,
        exit_tol: float = 1e-10,
    ) -> tuple:
        if not len(mu_0):
            mu = np.zeros(self.K_matrix.shape[1])
        else:
            mu = mu_0
        times = [0]
        supports = [np.sum(mu != 0)]
        objectives = [self.j(mu)]
        initial_time = time.perf_counter()
        k = 1
        while time.perf_counter() - initial_time < max_time and k <= max_iter:
            p_mu = -self.K_matrix.T @ self.residuum(mu)
            varphi = -p_mu + self.beta

            # Line search
            self.L = max(self.L * self.L_reduce_factor, self.L_min)
            descent_condition = False
            while not descent_condition:
                mu_plus, descent_condition = self.prox_step(mu, p_mu, varphi)
                if not descent_condition:
                    self.L = min(self.L * self.L_increase_factor, self.L_max)

            mu = mu_plus.copy()
            if np.any(mu <= -self.tol):
                logging.warning("Negative coefficients")

            # update metrics
            times.append(time.perf_counter() - initial_time)
            supports.append(np.sum(mu != 0))
            objectives.append(self.j(mu))

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
            f"L2 Prox Grad exited after {k} iterations and {times[-1]:.3f}s with final sparsity of {supports[-1]} and objective {objectives[-1]:.12E}"
        )
        return mu, objectives, times, supports
