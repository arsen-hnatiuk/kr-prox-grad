import numpy as np
import logging
import time
import cvxpy as cp
from typing import Callable
from itertools import combinations
import matplotlib.pyplot as plt
from lib.measure import Measure

logging.basicConfig(
    level=logging.DEBUG,
)


class KR_PROX_GRAD:

    def __init__(
        self,
        j: Callable,  # Objective
        p: Callable,  # Dual variable
        beta: float,
        domain: np.ndarray,
        L: float,
        wasserstein_weight: float = 1,
        tol: float = 1e-8,
    ) -> None:
        self.wasserstein_weight = wasserstein_weight
        self.j = j
        self.p = p
        self.beta = beta
        self.domain = domain
        self.L = L
        self.L_max = np.inf
        self.L_min = 1
        self.L_reduce_factor = 0.9
        self.L_increase_factor = 2
        self.tol = tol

    def compute_alphas_supports_y(self, y: np.array, varphi: np.array) -> dict:
        supports_dict = {}
        distances = np.linalg.norm(self.domain - y, axis=1).flatten()
        alpha = -np.min(varphi)
        o_j = varphi + alpha * self.wasserstein_weight * distances
        O_j = np.min(o_j)
        support_indices = np.where(o_j == O_j)[0]
        while True:
            # Compute the support on (alpha, alpha_+)
            support_distances = distances[support_indices].flatten()
            intermediate_support_indices = support_indices[
                np.where(support_distances == np.min(support_distances))[0]
            ]
            intermediate_support_distances = distances[
                intermediate_support_indices
            ].flatten()
            reference_index = intermediate_support_indices[0]
            supports_dict[alpha] = {
                "support_indices": support_indices,
                "support_distances": support_distances,
                "intermediate_support_indices": intermediate_support_indices,
                "intermediate_support_distances": intermediate_support_distances,
            }
            if len(intermediate_support_indices) == 1 and np.array_equal(
                self.domain[intermediate_support_indices[0]], y
            ):
                break

            # Determine alpha_+ and which points enter the support at alpha_+
            np.seterr(divide="ignore", invalid="ignore")
            intersection_vector = (varphi[reference_index] - varphi) / (
                self.wasserstein_weight * (distances - distances[reference_index])
            )
            np.seterr(divide="warn", invalid="warn")
            intersection_vector = np.nan_to_num(
                intersection_vector, nan=np.inf, posinf=np.inf, neginf=np.inf
            )
            intersection_vector[
                intersection_vector <= alpha * (1 + np.sign(alpha) * self.tol)
            ] = np.inf
            new_support_indices = np.where(
                intersection_vector == np.min(intersection_vector)
            )[
                0
            ]  # points that enter the support on alpha_+
            support_indices = np.hstack(
                (intermediate_support_indices, new_support_indices)
            )
            alpha_plus = np.min(intersection_vector)

            # Check if O_j intersects with alpha
            O_j_lower = (
                varphi[reference_index]
                + alpha * self.wasserstein_weight * distances[reference_index]
            )
            O_j_upper = (
                varphi[reference_index]
                + alpha_plus * self.wasserstein_weight * distances[reference_index]
            )
            if (
                O_j_lower <= alpha * (1 - np.sign(alpha) * self.tol)
                and O_j_upper >= alpha_plus * (1 + np.sign(alpha_plus) * self.tol)
            ) or (
                O_j_lower >= alpha * (1 + np.sign(alpha) * self.tol)
                and O_j_upper <= alpha_plus * (1 - np.sign(alpha_plus) * self.tol)
            ):
                if (
                    abs(1 - self.wasserstein_weight * distances[reference_index])
                    > self.tol
                ):
                    O_j_alpha = varphi[reference_index] / (
                        1 - self.wasserstein_weight * distances[reference_index]
                    )
                    supports_dict[O_j_alpha] = {
                        "support_indices": intermediate_support_indices,
                        "support_distances": intermediate_support_distances,
                        "intermediate_support_indices": intermediate_support_indices,
                        "intermediate_support_distances": intermediate_support_distances,
                    }  # Add alpha where O_j-alpha changes signs

            alpha = alpha_plus

        # Determine the last sign change breakpoint. |x-y|=0
        O_j = (
            varphi[reference_index]
            + alpha * self.wasserstein_weight * distances[reference_index]
        )
        if O_j >= alpha * (1 + np.sign(alpha) * self.tol):
            O_j_alpha = varphi[reference_index]
            supports_dict[O_j_alpha] = {
                "support_indices": intermediate_support_indices,
                "support_distances": intermediate_support_distances,
                "intermediate_support_indices": intermediate_support_indices,
                "intermediate_support_distances": intermediate_support_distances,
            }  # Add alpha where O_j-alpha changes signs

        return supports_dict

    def compute_alphas_supports(self, mu: Measure, varphi: np.ndarray) -> list:
        alphas = []
        all_supports_dict = (
            {}
        )  # {j: {alpha: {support indices:, distances:, intermediate support indices:, intermediate distances:}}}
        for j, y in enumerate(mu.support):
            supports_dict = self.compute_alphas_supports_y(y, varphi)
            all_supports_dict[j] = supports_dict
            alphas += list(supports_dict.keys())
        alphas = np.unique(alphas)
        return alphas, all_supports_dict

    def compute_kr_norm(self, mu_plus: Measure, mu: Measure) -> float:
        # ||mu_plus - mu||_{KR}
        mu_size = len(mu.coefficients)
        mu_plus_size = len(mu_plus.coefficients)
        variable_size = mu_plus_size * (mu_size + 1)

        # Build constraints matrix
        mu_constraint = np.zeros((mu_size, variable_size))
        mu_plus_constraint = np.zeros((mu_plus_size, variable_size))
        for j in range(mu_size):
            for i in range(mu_plus_size):
                mu_constraint[j, i * mu_size + j] = 1
                mu_plus_constraint[i, i * mu_size + j] = 1
                mu_plus_constraint[i, mu_plus_size * mu_size + i] = 1

        # Build coefficients for linear problem
        linear_term = np.zeros(variable_size)
        for j in range(mu_size):
            for i in range(mu_plus_size):
                linear_term[i * mu_size + j] = (
                    self.wasserstein_weight
                    * np.linalg.norm(mu.support[j] - mu_plus.support[i])
                    - 1
                )
        linear_term[-mu_plus_size:] = 1

        # Build linear problem
        Lambda = cp.Variable(variable_size)
        constraints = [
            Lambda >= 0,
            mu_constraint @ Lambda <= mu.coefficients,
            mu_plus_constraint @ Lambda == mu_plus.coefficients,
        ]
        problem = cp.Problem(
            cp.Minimize(linear_term.T @ Lambda),
            constraints,
        )
        problem.solve()
        solution = Lambda.value
        if problem.value == np.inf:
            logging.warning("Badly posed KR norm computation")

        kr_norm = linear_term @ solution + np.linalg.norm(mu.coefficients, ord=1)
        return kr_norm

    def compute_supports_per_j(
        self, alpha_lower: float, alpha_upper: float, all_supports_dict: dict
    ) -> dict:
        supports_per_j = {}  # {j: {support indices:, distances:}}
        for j, supports_per_alpha in all_supports_dict.items():
            incumbent_support_indices = np.array([])
            incumbent_distances = np.array([])
            for alpha_, inner_dict in supports_per_alpha.items():
                support_indices = inner_dict["support_indices"]
                support_distances = inner_dict["support_distances"]
                intermediate_support_indices = inner_dict[
                    "intermediate_support_indices"
                ]
                intermediate_support_distances = inner_dict[
                    "intermediate_support_distances"
                ]
                if abs(alpha_ - alpha_lower) <= abs(alpha_lower * self.tol) and abs(
                    alpha_ - alpha_upper
                ) <= abs(alpha_upper * self.tol):
                    incumbent_support_indices = support_indices.copy()
                    incumbent_distances = support_distances.copy()
                    break
                elif abs(alpha_ - alpha_lower) <= abs(alpha_lower * self.tol):
                    incumbent_support_indices = intermediate_support_indices.copy()
                    incumbent_distances = intermediate_support_distances.copy()
                    break
                elif alpha_ >= alpha_lower:
                    break
                incumbent_support_indices = intermediate_support_indices.copy()
                incumbent_distances = intermediate_support_distances.copy()
            supports_per_j[j] = {
                "support_indices": incumbent_support_indices,
                "distances": incumbent_distances,
            }
        return supports_per_j

    def kr_subproblem(self, mu: Measure, varphi: np.ndarray) -> tuple:
        alphas, all_supports_dict = self.compute_alphas_supports(mu, varphi)
        alphas = np.append(alphas, np.inf)
        alpha_intervals = []
        for i, alpha in enumerate(alphas[:-1]):
            alpha_intervals.append((alpha, alpha))
            alpha_intervals.append((alpha, alphas[i + 1]))

        # Loop over the alphas in increasing order to find solution
        kr_norm_lower = np.inf
        mu_kr_lower = Measure()
        number_transported_points_lower = 0
        for i, (alpha_lower, alpha_upper) in enumerate(alpha_intervals):
            if alpha_lower == alpha_upper:
                alpha = alpha_lower
                kr_norm_upper = kr_norm_lower
                mu_kr_upper = mu_kr_lower.copy()
                number_transported_points_upper = number_transported_points_lower
                supports_per_j = self.compute_supports_per_j(
                    alpha, alpha, all_supports_dict
                )
                kr_norm_lower = np.linalg.norm(mu.coefficients, ord=1)
                number_transported_points_lower = 0
                mu_kr_lower = Measure()
                for j, inner_dict in supports_per_j.items():
                    support_indices = inner_dict["support_indices"]
                    reference_index = support_indices[0]
                    distances = inner_dict["distances"]
                    O_j = (
                        varphi[reference_index]
                        + self.wasserstein_weight * alpha * distances[0]
                    )
                    if O_j >= alpha * (1 + np.sign(alpha) * self.tol):
                        pass
                    else:
                        min_dist = np.min(distances)
                        if not (
                            abs(O_j - alpha) < abs(alpha * self.tol)
                            and min_dist >= 1 / self.wasserstein_weight
                        ):
                            min_dist_index = support_indices[np.argmin(distances)]
                            tau_j = mu.coefficients[j]
                            kr_norm_lower += tau_j * (
                                self.wasserstein_weight * min_dist - 1
                            )
                            mu_kr_lower += Measure(
                                support=[self.domain[min_dist_index]],
                                coefficients=[tau_j],
                            )
                            if min_dist > 0:
                                number_transported_points_lower += 1
                if i == 0:  # alpha=-min varphi
                    # logging.info(f"alpha: {alpha}, KR-: {kr_norm_lower*self.L}")
                    if (
                        kr_norm_lower
                        <= alpha * (1 + np.sign(alpha) * self.tol) / self.L
                    ):
                        # There exists a solution, construct by choosing optimal u
                        kr_norm = kr_norm_lower
                        mu_kr = mu_kr_lower.copy()
                        number_transported_points = number_transported_points_lower
                        u = max(alpha / self.L - kr_norm, 0)
                        x_bar = np.argmin(varphi)
                        mu_kr += Measure(support=[self.domain[x_bar]], coefficients=[u])
                        return mu_kr, alpha / self.L, number_transported_points
                else:
                    # logging.info(
                    #     f"alpha-: {alpha}, alpha+: {alpha}, KR-: {kr_norm_lower*self.L}, KR+: {kr_norm_upper*self.L}"
                    # )
                    if (
                        kr_norm_lower
                        <= alpha * (1 + np.sign(alpha) * self.tol) / self.L
                        and kr_norm_upper
                        >= alpha * (1 - np.sign(alpha) * self.tol) / self.L
                    ):
                        # There exists a valid solution: construct by convex composition
                        if abs(kr_norm_lower - alpha / self.L) < abs(alpha * self.tol):
                            number_transported_points = number_transported_points_lower
                        else:
                            number_transported_points = number_transported_points_upper
                        if abs(kr_norm_lower - kr_norm_upper) < abs(alpha * self.tol):
                            return (
                                mu_kr_lower,
                                alpha / self.L,
                                number_transported_points,
                            )
                        else:
                            theta = min(
                                max(
                                    (alpha / self.L - kr_norm_upper)
                                    / (kr_norm_lower - kr_norm_upper),
                                    0,
                                ),
                                1,
                            )
                            mu_kr = mu_kr_lower * theta + mu_kr_upper * (1 - theta)
                            return mu_kr, alpha / self.L, number_transported_points
            else:
                # (alpha-, alpha+) is a true interval
                kr_norm = kr_norm_lower
                mu_kr = mu_kr_lower.copy()
                number_transported_points = number_transported_points_lower
                # logging.info(
                #     f"alpha-: {alpha_lower}, alpha+: {alpha_upper}, KR-: {kr_norm*self.L}, KR+: {kr_norm*self.L}"
                # )
                if (
                    kr_norm
                    <= alpha_upper * (1 + np.sign(alpha_upper) * self.tol) / self.L
                    and kr_norm
                    >= alpha_lower * (1 - np.sign(alpha_lower) * self.tol) / self.L
                ):
                    return mu_kr, kr_norm, number_transported_points
        logging.warning("No KR solution found")

    def kr_step(
        self, mu: Measure, p_mu: Callable, varphi: np.array, log_results: bool
    ) -> tuple:
        descent_condition = True
        if not len(mu.coefficients):
            # Reference measure is null
            position = np.argmin(varphi)
            coef = -varphi[position] / self.L
            return (
                Measure(support=[self.domain[position]], coefficients=[coef]),
                descent_condition,
                0,
                coef * self.L,
            )
        else:
            mu_plus, kr_norm, number_transported_points = self.kr_subproblem(mu, varphi)

            if log_results:
                kr_norm_cvx = self.compute_kr_norm(mu_plus, mu)
                if np.abs(kr_norm - kr_norm_cvx) > 1e-6:
                    logging.warning("Mismatch in KR norm computation")

            # Check KR descent:
            diff = self.j(mu_plus) - self.j(mu)
            kr_rhs = (
                -mu_plus.duality_pairing(p_mu)
                + self.beta * np.linalg.norm(mu_plus.coefficients, ord=1)
                + mu.duality_pairing(p_mu)
                - self.beta * np.linalg.norm(mu.coefficients, ord=1)
                + 0.5 * self.L * kr_norm**2
            )
            if diff > kr_rhs or diff > 0:
                descent_condition = False
            return (
                mu_plus,
                descent_condition,
                number_transported_points,
                kr_norm * self.L,
            )

    def solve(
        self,
        mu_0: Measure = Measure(),
        max_time: float = 60.0,
        max_iter: int = 250,
        log_results: bool = True,
        optimum: float = np.inf,
        exit_tol: float = 1e-10,
    ) -> tuple:
        mu = mu_0
        times = [0]
        supports = [len(mu.support)]
        objectives = [self.j(mu)]
        initial_time = time.perf_counter()
        k = 1
        while time.perf_counter() - initial_time < max_time and k <= max_iter:
            p_mu = self.p(mu)
            varphi = -p_mu(self.domain) + self.beta

            # Line search
            self.L = max(self.L * self.L_reduce_factor, self.L_min)
            descent_condition = False
            while not descent_condition:
                mu_plus, descent_condition, number_transported_points, alpha = (
                    self.kr_step(mu, p_mu, varphi, log_results)
                )
                if not descent_condition:
                    self.L = min(self.L * self.L_increase_factor, self.L_max)

            mu = mu_plus.copy()
            if np.any(mu.coefficients <= -self.tol):
                logging.warning("Negative coefficients")

            # update metrics
            times.append(time.perf_counter() - initial_time)
            supports.append(len(mu.support))
            objectives.append(self.j(mu))

            if objectives[-1] > objectives[-2] * (1 + self.tol):
                logging.warning("Ascent observed")
            if log_results:
                logging.info(
                    f"{k}: L:{self.L:.3E}, transported points: {number_transported_points}, alpha: {alpha:.3E}, support {supports[-1]}, objective: {objectives[-1]:.12E}"
                )
            # Check optimality
            if abs(objectives[-1] - optimum) < exit_tol:
                break
            k += 1

            # if k==2500:
            #     logging.basicConfig(
            #         level=logging.ERROR,
            #     )
            #     B, D = np.meshgrid(
            #                 *(np.linspace(0, 1, 100 + 2)[1:-1] for _ in range(2))
            #             )
            #     vals = varphi.reshape((100, 100))
            #     plt.contourf(B, D, vals, levels=100)
            #     plt.colorbar()
            #     # for i, x in enumerate(true_sources):
            #     #     if i:
            #     #         plt.plot([x[0]], [x[1]], "P", c="r", markersize=10)
            #     #     else:
            #     #         plt.plot([x[0]], [x[1]], "P", c="r", markersize=10, label="True sources")
            #     for i, x in enumerate(mu.support):
            #         if i:
            #             plt.plot([x[0]], [x[1]], "o", c="b")
            #         else:
            #             plt.plot([x[0]], [x[1]], "o", c="b", label="Predicted support")
            #     plt.legend()
            #     plt.show()
            #     logging.basicConfig(
            #         level=logging.DEBUG,
            #     )
        logging.info(
            f"KR Prox Grad exited after {k} iterations and {times[-1]:.3f}s with final sparsity of {supports[-1]} and objective {objectives[-1]:.12E}"
        )
        return mu, objectives, times, supports
