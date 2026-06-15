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
        K_transpose: np.ndarray,
        beta: float,
        L: float,
    ) -> None:
        self.K_transpose = K_transpose
        self.target = target
        self.f = (
            lambda mu: 0.5
            * np.linalg.norm(mu.duality_pairing(self.K_transpose) - target) ** 2
        )
        self.residuum = lambda mu: mu.duality_pairing(self.K_transpose) - self.target
        self.beta = beta
        self.g = lambda mu: beta * np.linalg.norm(mu.coefficients, ord=1)
        self.j = lambda mu: self.f(mu) + self.g(mu)
        self.L = L
        self.L_max = np.inf
        self.L_min = 0
        self.L_reduce_factor = 0.9
        self.L_increase_factor = 2
        self.M = self.j(self.u_0) / self.beta  # Bound on the norm of iterates
        self.machine_precision = 1e-12

    def frank_wolfe_step(self, mu: Measure, p_mu: Callable, v: Measure) -> tuple:
        g = v.duality_pairing(p_mu)
        norm_v = np.linalg.norm(v, ord=1)
        step_size = g / (self.L * norm_v**2)
        mu_plus = mu + v * step_size

        # Check descent
        descent_condition = True
        diff = self.j(mu_plus) - self.j(mu)
        descent_rhs = (
            -step_size * g
            + self.g(mu_plus)
            - self.g(mu)
            + 0.5 * step_size**2 * norm_v**2
        )
        if diff > descent_rhs or diff > 0:
            descent_condition = False
        return mu_plus, descent_condition

    def solve(
        self,
        max_time: float = 60.0,
        max_iter: int = 250,
        log_results: bool = True,
        optimum: float = np.inf,
        exit_tol: float = 1e-10,
        mu_0: Measure = Measure(),
    ) -> tuple:
        mu = mu_0.copy()
        k = 1
        initial_time = time.perf_counter()
        objectives = [self.j(mu)]
        times = [time.perf_counter() - initial_time]
        supports = [0]
        while time.perf_counter() - initial_time < max_time and k <= max_iter:
            residuum_u = self.residuum(mu)
            p_mu = -self.K_transpose @ residuum_u
            x = np.argmax(np.abs(p_mu))
            v = Measure(support=[[x]], coefficients=[self.M * np.sign(p_mu[x])])

            # Line search
            self.L = max(self.L * self.L_reduce_factor, self.L_min)
            descent_condition = False
            while not descent_condition:
                mu_plus, descent_condition = self.frank_wolfe_step(mu, p_mu, v)
                if not descent_condition:
                    self.L = min(self.L * self.L_increase_factor, self.L_max)

            mu = mu_plus.copy()
            if np.any(mu <= -self.tol):
                logging.warning("Negative coefficients")

            self.M = self.j(mu) / self.beta

            objectives.append(self.j(mu))
            times.append(time.perf_counter() - initial_time)
            supports.append(len(mu.support))

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
        return u, objectives, times, supports
