import numpy as np
import logging
import time
import cvxpy as cp
from typing import Callable
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
    ) -> None:
        self.j = j
        self.p = p
        self.beta = beta
        self.domain = domain
        self.L = L

    def compute_alphas_supports_y(self, y: np.array, varphi: np.array) -> dict:
        supports_dict = {}
        distances = np.linalg.norm(self.domain - y, axis=1).flatten()
        alpha = -np.min(varphi)
        o_j = varphi + alpha * distances
        support_indices = np.where(o_j == np.min(o_j))[0]
        while True:
            # Compute the support on (alpha, alpha_+)
            support_distances = distances[support_indices].flatten()
            intermediate_support_indices = support_indices[
                np.where(support_distances == np.min(support_distances))[0]
            ]
            intermediate_support_distances = distances[
                intermediate_support_indices
            ].flatten()
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
            reference_index = intermediate_support_indices[0]
            np.seterr(divide="ignore", invalid="ignore")
            intersection_vector = (varphi[reference_index] - varphi) / (
                distances - distances[reference_index]
            )
            np.seterr(divide="warn", invalid="warn")
            intersection_vector = np.nan_to_num(
                intersection_vector, nan=np.inf, posinf=np.inf, neginf=np.inf
            )
            intersection_vector[intersection_vector <= alpha + 1e-12] = np.inf
            new_support_indices = np.where(
                intersection_vector == np.min(intersection_vector)
            )[
                0
            ]  # points that enter the support on alpha_+
            support_indices = np.hstack(
                (intermediate_support_indices, new_support_indices)
            )
            alpha = np.min(intersection_vector)  # alpha_+
        return supports_dict

    def compute_alphas_supports(self, u: Measure, varphi: np.ndarray) -> list:
        alphas = []
        all_supports_dict = (
            {}
        )  # {j: {alpha: {support indices:, distances:, intermediate support indices:, intermediate distances:}}}
        for j, y in enumerate(u.support):
            supports_dict = self.compute_alphas_supports_y(y, varphi)
            all_supports_dict[j] = supports_dict
            alphas += list(supports_dict.keys())
        alphas = np.unique(alphas)
        return alphas, all_supports_dict

    def screen_alpha(
        self,
        u: Measure,
        varphi: np.ndarray,
        alpha_bounds: tuple,
        supports_per_j: dict,
    ) -> bool:
        # Check if the given alpha can contain a candidate
        lower_bound = np.linalg.norm(u.coefficients, ord=1)
        if alpha_bounds[0] == alpha_bounds[1] == -np.min(varphi):
            upper_bound = np.inf
        else:
            upper_bound = np.linalg.norm(u.coefficients, ord=1)
        for j, inner_dict in supports_per_j.items():
            distances = inner_dict["distances"].copy()
            min_distance = np.min(distances)
            lower_bound += u.coefficients[j] * min((min_distance - 1), 0)
            max_distance = np.max(distances)
            upper_bound += u.coefficients[j] * max((max_distance - 1), 0)
        if (
            lower_bound <= alpha_bounds[1] / self.L
            and alpha_bounds[0] / self.L <= upper_bound
        ):
            return True
        else:
            return False

    def quadratic_problem(
        self,
        u: Measure,
        varphi: np.ndarray,
        supports_per_j: dict,
        log_results: bool = False,
    ) -> list:
        total_support_indices = np.unique(
            np.hstack(
                [d["support_indices"] for d in supports_per_j.values()]
            ).flatten(),
        )
        total_support_size = len(total_support_indices)
        variable_size = total_support_size * len(u.coefficients) + 1
        u_norm = np.linalg.norm(u.coefficients, ord=1)
        x_bar = np.argmin(varphi)
        min_varphi = varphi[x_bar]

        # Build coefficients for quadratic problem
        varphi_vector = np.zeros(variable_size)
        distance_vector = np.zeros(variable_size)
        support_indices_per_j = {}
        for j, inner_dict in supports_per_j.items():
            inner_support_indices = inner_dict["support_indices"]
            inner_distances = inner_dict["distances"] - 1
            inner_varphi = varphi[inner_support_indices]
            inner_indices = []
            for sup, dist, vphi in zip(
                inner_support_indices, inner_distances, inner_varphi
            ):
                ind = np.where(total_support_indices == sup)[0][0]
                inner_indices.append(ind)
                ell_j_index = ind * len(u.coefficients) + j
                varphi_vector[ell_j_index] = vphi
                distance_vector[ell_j_index] = dist
            support_indices_per_j[j] = np.array(inner_indices)
        varphi_vector[-1] = min_varphi
        distance_vector[-1] = 1
        quadratic_term = np.sqrt(self.L) * distance_vector
        linear_term = varphi_vector + self.L * u_norm * distance_vector

        # Build constraints matrices
        null_constraint = np.zeros((variable_size, variable_size))
        sum_constraint = np.zeros((len(u.coefficients), variable_size))
        for j, inner_support_indices in support_indices_per_j.items():
            for sup in inner_support_indices:
                sum_constraint[j, sup * len(u.coefficients) + j] = 1
            for ell in range(total_support_size):
                if ell not in inner_support_indices:
                    null_constraint[
                        ell * len(u.coefficients) + j, ell * len(u.coefficients) + j
                    ] = 1

        # Build quadratic problem
        Lambda = cp.Variable(variable_size)
        constraints = [
            Lambda >= 0,
            sum_constraint @ Lambda <= u.coefficients,
            null_constraint @ Lambda == 0,
        ]
        problem = cp.Problem(
            cp.Minimize(
                0.5 * cp.square(quadratic_term.T @ Lambda) + linear_term.T @ Lambda
            ),
            constraints,
        )
        try:
            problem.solve()
        except cp.error.DCPError:
            logging.info(min_varphi)
            logging.info(varphi_vector)
            logging.info(distance_vector)
            logging.info(null_constraint)
            logging.info(sum_constraint)
            problem.solve(verbose=True)
        solution = Lambda.value

        # Reconstruct KR norm from solution
        kr_norm = distance_vector @ solution + u_norm

        # Reconstruct measure from solution
        u_plus = Measure(support=[self.domain[x_bar]], coefficients=[solution[-1]])
        for ell in range(total_support_size):
            for j in range(len(u.coefficients)):
                coef = solution[ell * len(u.coefficients) + j]
                if coef:
                    u_plus += Measure(
                        support=[self.domain[total_support_indices[ell]]],
                        coefficients=[coef],
                    )

        return u_plus, kr_norm

    def kr_step(
        self, u: Measure, p_u: Callable, varphi: np.array, log_results: bool
    ) -> Measure:
        if not len(u.coefficients):
            # Reference measure is null
            position = np.argmin(varphi)
            coef = -varphi[position] / self.L
            return Measure(support=[self.domain[position]], coefficients=[coef])
        else:
            alphas, all_supports_dict = self.compute_alphas_supports(u, varphi)
            alphas = np.append(alphas, np.inf)
            if log_results:
                logging.info(f"alphas: {alphas}")
            for i, alpha in enumerate(alphas[:-1]):
                # Loop over the alphas in increasing order to find solution
                for alpha_lower, alpha_upper in [
                    (alpha, alpha),
                    (alpha, alphas[i + 1]),
                ]:
                    # Check in alpha and (alpha, alpha_+)
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
                            incumbent_support_indices = (
                                intermediate_support_indices.copy()
                            )
                            incumbent_distances = intermediate_support_distances.copy()
                            if alpha_ == alpha_lower and alpha_ == alpha_upper:
                                incumbent_support_indices = support_indices.copy()
                                incumbent_distances = support_distances.copy()
                                break
                            elif alpha_ >= alpha_lower:
                                break
                        supports_per_j[j] = {
                            "support_indices": incumbent_support_indices,
                            "distances": incumbent_distances,
                        }

                    # logging.info(supports_per_j)
                    if not self.screen_alpha(
                        u, varphi, (alpha_lower, alpha_upper), supports_per_j
                    ):
                        if log_results:
                            logging.info("alpha failed screening")
                        continue

                    u_plus, kr_norm = self.quadratic_problem(u, varphi, supports_per_j)
                    if log_results:
                        logging.info(
                            f"({alpha_lower/self.L}, {alpha_upper/self.L}): {kr_norm}"
                        )
                    if (
                        alpha_lower / self.L - 1e-6 <= kr_norm
                        and alpha_upper / self.L + 1e-6 >= kr_norm
                    ):
                        # Check KR descent:
                        diff = self.j(u_plus) - self.j(u)
                        rk_rhs = (
                            -u_plus.duality_pairing(p_u)
                            + self.beta * np.linalg.norm(u.coefficients, ord=1)
                            + u.duality_pairing(p_u)
                            - self.beta * np.linalg.norm(u.coefficients, ord=1)
                            + 0.5 * self.L * kr_norm**2
                        )
                        if log_results:
                            logging.info(
                                f"KR Descent condition satisfied: {diff<= rk_rhs}"
                            )
                        return u_plus
        logging.warning("NO KR SOLUTION FOUND")
        return u_plus

    def solve(
        self,
        u_0: Measure = Measure(),
        max_time: float = 60.0,
        log_results: bool = True,
    ) -> tuple:
        u = u_0
        times = [0]
        supports = [len(u.support)]
        objectives = [self.j(u)]
        initial_time = time.perf_counter()
        k = 1
        while time.perf_counter() - initial_time < max_time:
            p_u = self.p(u)
            varphi = -p_u(self.domain) + self.beta
            if np.min(varphi) >= 0:
                # Reached optimality
                break
            u = self.kr_step(u, p_u, varphi, log_results)

            # update metrics
            times.append(time.perf_counter() - initial_time)
            supports.append(len(u.support))
            objectives.append(self.j(u))

            if log_results:
                logging.info(
                    f"{k}: support {supports[-1]}, objective: {objectives[-1]:.12E}"
                )
            k += 1
        logging.info(
            f"KR Prox Grad exited after {k} iterations and {times[-1]:.3f}s with final sparsity of {supports[-1]} and objective {objectives[-1]:.12E}"
        )
        return u, objectives, times, supports
