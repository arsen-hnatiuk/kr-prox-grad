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
        self.L_max = 2 * L
        self.L_min = 1
        self.L_reduce_factor = 0.75
        self.L_increase_factor = 2
        self.wasserstein_weight = 1

    def compute_alphas_supports_y(self, y: np.array, varphi: np.array) -> dict:
        supports_dict = {}
        distances = np.linalg.norm(self.domain - y, axis=1).flatten()
        alpha = -np.min(varphi)
        alpha_plus = alpha
        o_j = varphi + alpha * distances
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
            alpha_plus = np.min(intersection_vector)  # alpha_+

            # Check if O_j intersects with alpha
            O_j_lower = varphi[reference_index] + alpha * distances[reference_index]
            O_j_upper = (
                varphi[reference_index] + alpha_plus * distances[reference_index]
            )
            # logging.info(
            #     f"alpha: {alpha}, alpha+: {alpha_plus}, O-: {O_j_lower}, O+:{O_j_upper}"
            # )
            if (O_j_lower < alpha and O_j_upper > alpha_plus) or (
                O_j_lower > alpha and O_j_upper < alpha_plus
            ):
                O_j_alpha = varphi[reference_index] / (1 - distances[reference_index])
                # logging.info("FOUND OJOJOJOJO")
                # logging.info(O_j_alpha)
                supports_dict[O_j_alpha] = {
                    "support_indices": intermediate_support_indices,
                    "support_distances": intermediate_support_distances,
                    "intermediate_support_indices": intermediate_support_indices,
                    "intermediate_support_distances": intermediate_support_distances,
                }  # Add alpha where O_j-alpha changes signs
            alpha = alpha_plus
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

    # def screen_alpha(
    #     self,
    #     u: Measure,
    #     varphi: np.ndarray,
    #     alpha_bounds: tuple,
    #     supports_per_j: dict,
    # ) -> bool:
    #     # Check if the given alpha can contain a candidate
    #     lower_bound = np.linalg.norm(u.coefficients, ord=1)
    #     if alpha_bounds[0] == alpha_bounds[1] == -np.min(varphi):
    #         upper_bound = np.inf
    #     else:
    #         upper_bound = np.linalg.norm(u.coefficients, ord=1)
    #     for j, inner_dict in supports_per_j.items():
    #         suport_indices = inner_dict["support_indices"]
    #         distances = inner_dict["distances"].copy()
    #         min_distance = np.min(distances)
    #         lower_bound += u.coefficients[j] * min((min_distance - 1), 0)
    #         max_distance = np.max(distances)
    #         upper_bound += u.coefficients[j] * max((max_distance - 1), 0)
    #     if (
    #         lower_bound <= alpha_bounds[1] / self.L
    #         and alpha_bounds[0] / self.L <= upper_bound
    #     ):
    #         return True
    #     else:
    #         return False

    # def quadratic_problem(
    #     self,
    #     u: Measure,
    #     varphi: np.ndarray,
    #     supports_per_j: dict,
    #     log_results: bool = False,
    # ) -> list:
    #     total_support_indices = np.unique(
    #         np.hstack(
    #             [d["support_indices"] for d in supports_per_j.values()]
    #         ).flatten(),
    #     )
    #     total_support_size = len(total_support_indices)
    #     variable_size = total_support_size * len(u.coefficients) + 1
    #     u_norm = np.linalg.norm(u.coefficients, ord=1)
    #     x_bar = np.argmin(varphi)
    #     min_varphi = varphi[x_bar]

    #     # Build coefficients for quadratic problem
    #     varphi_vector = np.zeros(variable_size)
    #     distance_vector = np.zeros(variable_size)
    #     support_indices_per_j = {}
    #     for j, inner_dict in supports_per_j.items():
    #         inner_support_indices = inner_dict["support_indices"]
    #         inner_distances = inner_dict["distances"] - 1
    #         inner_varphi = varphi[inner_support_indices]
    #         inner_indices = []
    #         for sup, dist, vphi in zip(
    #             inner_support_indices, inner_distances, inner_varphi
    #         ):
    #             ind = np.where(total_support_indices == sup)[0][0]
    #             inner_indices.append(ind)
    #             ell_j_index = ind * len(u.coefficients) + j
    #             varphi_vector[ell_j_index] = vphi
    #             distance_vector[ell_j_index] = dist
    #         support_indices_per_j[j] = np.array(inner_indices)
    #     varphi_vector[-1] = min_varphi
    #     distance_vector[-1] = 1
    #     quadratic_term = np.sqrt(self.L) * distance_vector
    #     linear_term = varphi_vector + self.L * u_norm * distance_vector

    #     # Build constraints matrices
    #     null_constraint = np.zeros((variable_size, variable_size))
    #     sum_constraint = np.zeros((len(u.coefficients), variable_size))
    #     for j, inner_support_indices in support_indices_per_j.items():
    #         for sup in inner_support_indices:
    #             sum_constraint[j, sup * len(u.coefficients) + j] = 1
    #         for ell in range(total_support_size):
    #             if ell not in inner_support_indices:
    #                 null_constraint[
    #                     ell * len(u.coefficients) + j, ell * len(u.coefficients) + j
    #                 ] = 1

    #     # Build quadratic problem
    #     Lambda = cp.Variable(variable_size)
    #     constraints = [
    #         Lambda >= 0,
    #         sum_constraint @ Lambda <= u.coefficients,
    #         null_constraint @ Lambda == 0,
    #     ]
    #     problem = cp.Problem(
    #         cp.Minimize(
    #             0.5 * cp.square(quadratic_term.T @ Lambda) + linear_term.T @ Lambda
    #         ),
    #         constraints,
    #     )
    #     try:
    #         problem.solve(eps_rel=1e-10)
    #     except cp.error.DCPError:
    #         logging.info(min_varphi)
    #         logging.info(varphi_vector)
    #         logging.info(distance_vector)
    #         logging.info(null_constraint)
    #         logging.info(sum_constraint)
    #         problem.solve(verbose=True)
    #     solution = Lambda.value
    #     # solution[solution < 0] = 0
    #     # if log_results:
    #     #     logging.info(x_bar)
    #     #     logging.info(solution[-1])

    #     # Check number of transported points
    #     number_transported_points = 0
    #     for j, inner_support_indices in support_indices_per_j.items():
    #         transport_vector = []
    #         for ell in range(total_support_size):
    #             transport_vector.append(solution[ell * len(u.coefficients) + j])
    #         transport_vector = np.array(transport_vector)
    #         transport_mass = np.sum(transport_vector)
    #         j_point = u.support[j]
    #         j_index_global = np.where(
    #             np.linalg.norm(self.domain - j_point, axis=1) == 0
    #         )[0][0]
    #         j_index_local = np.where(total_support_indices == j_index_global)[0]
    #         if len(j_index_local):
    #             reference_vector = (
    #                 transport_mass * np.eye(1, total_support_size, j_index_local[0])[0]
    #             )
    #         else:
    #             reference_vector = np.zeros(total_support_size)
    #         transport_difference = transport_vector - reference_vector
    #         number_transported_points += np.sum(transport_difference != 0)

    #     # Reconstruct KR norm from solution
    #     kr_norm = distance_vector @ solution + u_norm

    #     # Reconstruct measure from solution
    #     u_plus = Measure(support=[self.domain[x_bar]], coefficients=[solution[-1]])
    #     for ell in range(total_support_size):
    #         for j in range(len(u.coefficients)):
    #             coef = solution[ell * len(u.coefficients) + j]
    #             if coef:
    #                 u_plus += Measure(
    #                     support=[self.domain[total_support_indices[ell]]],
    #                     coefficients=[coef],
    #                 )

    #     return u_plus, kr_norm, number_transported_points

    # def kr_step_old(
    #     self, u: Measure, p_u: Callable, varphi: np.array, log_results: bool
    # ) -> tuple:
    #     descent_condition = True
    #     if not len(u.coefficients):
    #         # Reference measure is null
    #         position = np.argmin(varphi)
    #         coef = -varphi[position] / self.L
    #         return (
    #             Measure(support=[self.domain[position]], coefficients=[coef]),
    #             descent_condition,
    #             0,
    #         )
    #     else:
    #         alphas, all_supports_dict = self.compute_alphas_supports(u, varphi)
    #         alphas = np.append(alphas, np.inf)
    #         # if log_results:
    #         #     logging.info(f"alphas: {alphas}")
    #         found_us = []
    #         for i, alpha in enumerate(alphas[:-1]):
    #             # Loop over the alphas in increasing order to find solution
    #             for alpha_lower, alpha_upper in [
    #                 (alpha, alpha),
    #                 (alpha, alphas[i + 1]),
    #             ]:
    #                 # Check in alpha and (alpha, alpha_+)
    #                 supports_per_j = {}  # {j: {support indices:, distances:}}
    #                 for j, supports_per_alpha in all_supports_dict.items():
    #                     incumbent_support_indices = np.array([])
    #                     incumbent_distances = np.array([])
    #                     for alpha_, inner_dict in supports_per_alpha.items():
    #                         support_indices = inner_dict["support_indices"]
    #                         support_distances = inner_dict["support_distances"]
    #                         intermediate_support_indices = inner_dict[
    #                             "intermediate_support_indices"
    #                         ]
    #                         intermediate_support_distances = inner_dict[
    #                             "intermediate_support_distances"
    #                         ]
    #                         incumbent_support_indices = (
    #                             intermediate_support_indices.copy()
    #                         )
    #                         incumbent_distances = intermediate_support_distances.copy()
    #                         if alpha_ == alpha_lower and alpha_ == alpha_upper:
    #                             incumbent_support_indices = support_indices.copy()
    #                             incumbent_distances = support_distances.copy()
    #                             break
    #                         elif alpha_ >= alpha_lower:
    #                             break
    #                     supports_per_j[j] = {
    #                         "support_indices": incumbent_support_indices,
    #                         "distances": incumbent_distances,
    #                     }

    #                 if not self.screen_alpha(
    #                     u, varphi, (alpha_lower, alpha_upper), supports_per_j
    #                 ):
    #                     # if log_results:
    #                     #     logging.info(
    #                     #         f"alphas {(alpha_lower, alpha_upper)} failed screening"
    #                     #     )
    #                     continue

    #                 u_plus, kr_norm, number_transported_points = self.quadratic_problem(
    #                     u, varphi, supports_per_j, log_results=log_results
    #                 )
    #                 found_us.append(u_plus.copy())
    #                 # if log_results:
    #                 #     logging.info(
    #                 #         f"({alpha_lower/self.L}, {alpha_upper/self.L}): {kr_norm}"
    #                 #     )
    #                 if (
    #                     alpha_lower / self.L - 1e-6 <= kr_norm
    #                     and alpha_upper / self.L + 1e-6 >= kr_norm
    #                 ):
    #                     # Check KR descent:
    #                     diff = self.j(u_plus) - self.j(u)
    #                     kr_rhs = (
    #                         -u_plus.duality_pairing(p_u)
    #                         + self.beta * np.linalg.norm(u.coefficients, ord=1)
    #                         + u.duality_pairing(p_u)
    #                         - self.beta * np.linalg.norm(u.coefficients, ord=1)
    #                         + 0.5 * self.L * kr_norm**2
    #                     )
    #                     if log_results and diff > kr_rhs:
    #                         logging.warning(f"KR Descent condition failed")
    #                         descent_condition = False
    #                     return u_plus, descent_condition, number_transported_points
    #     if log_results:
    #         logging.warning("No KR solution found")
    #     us_values = [self.j(u_) for u_ in found_us]
    #     return (
    #         found_us[np.argmin(us_values)],
    #         descent_condition,
    #         number_transported_points,
    #     )

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
                mu_constraint[j, i * mu_plus_size + j] = 1
                mu_plus_constraint[i, i * mu_plus_size + j] = 1
                mu_plus_constraint[i, mu_plus_size * mu_size + i] = 1

        # Build coefficients for linear problem
        linear_term = np.zeros(variable_size)
        for j in range(mu_size):
            for i in range(mu_plus_size):
                linear_term[i * mu_plus_size + j] = (
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
            mu_plus_constraint @ Lambda <= mu_plus.coefficients,
        ]
        problem = cp.Problem(
            cp.Minimize(linear_term.T @ Lambda),
            constraints,
        )
        problem.solve()
        solution = Lambda.value

        kr_norm = linear_term @ solution + np.linalg.norm(mu.coefficients, ord=1)
        return kr_norm

    def kr_subproblem_first_alpha(
        self,
        alpha: float,
        mu: Measure,
        varphi: np.ndarray,
        supports_per_j: dict,
        log_results: bool,
    ) -> tuple:
        x_bar = np.argmin(varphi)
        min_varphi = varphi[x_bar]
        if alpha != -min_varphi:
            logging.warning("Erroneous first alpha calculation")
        success = False
        number_transported_points = 0
        kr_norm = np.linalg.norm(mu.coefficients, ord=1)
        mu_kr = Measure()
        for j, inner_dict in supports_per_j.items():
            support_indices = inner_dict["support_indices"]
            reference_index = support_indices[0]
            distances = inner_dict["distances"]
            O_j = (
                varphi[reference_index] + self.wasserstein_weight * alpha * distances[0]
            )
            if O_j > alpha:
                pass
            else:
                min_dist = np.min(distances)
                if not (O_j == alpha and min_dist >= 1):
                    min_dist_index = support_indices[np.argmin(distances)]
                    tau_j = mu.coefficients[j]
                    kr_norm += tau_j * (self.wasserstein_weight * min_dist - 1)
                    mu_kr += Measure(
                        support=[self.domain[min_dist_index]], coefficients=[tau_j]
                    )
                    if min_dist > 0:
                        number_transported_points += 1
        if kr_norm * self.L > alpha:
            return mu_kr, kr_norm, success, number_transported_points

        # There exists a solution, construct by choosing optimal u
        success = True
        u = max(alpha / self.L - kr_norm, 0)
        mu_kr += Measure(support=[self.domain[x_bar]], coefficients=[u])
        return mu_kr, alpha / self.L, success, number_transported_points

    def kr_subproblem(
        self,
        alpha_lower: float,
        alpha_upper: float,
        mu: Measure,
        varphi: np.ndarray,
        supports_per_j: dict,
        log_results: bool,
    ) -> tuple:
        success = False
        number_transported_points_lower = 0
        number_transported_points_upper = 0
        if alpha_upper > 1e100:
            # alpha_upper = inf
            alpha = 1e100
        else:
            alpha = 0.5 * alpha_lower + 0.5 * alpha_upper
        kr_norm_lower = np.linalg.norm(mu.coefficients, ord=1)
        kr_norm_upper = np.linalg.norm(mu.coefficients, ord=1)
        mu_kr_lower = Measure()
        mu_kr_upper = Measure()
        for j, inner_dict in supports_per_j.items():
            support_indices = inner_dict["support_indices"]
            reference_index = support_indices[0]
            distances = inner_dict["distances"]
            O_j = (
                varphi[reference_index] + self.wasserstein_weight * alpha * distances[0]
            )  # O_j>alpha is constant on (alpha-, alpha+) by construction
            if O_j > alpha:
                pass
            else:
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                # if min_dist > 0 or max_dist > 0:
                #     logging.info(f"{j}: min dist: {min_dist}, max dist: {max_dist}")
                min_dist_index = support_indices[np.argmin(distances)]
                max_dist_index = support_indices[np.argmax(distances)]
                tau_j = mu.coefficients[j]
                if not (O_j == alpha and min_dist >= 1):
                    kr_norm_lower += tau_j * (self.wasserstein_weight * min_dist - 1)
                    mu_kr_lower += Measure(
                        support=[self.domain[min_dist_index]], coefficients=[tau_j]
                    )
                    if min_dist > 0:
                        number_transported_points_lower += 1
                if not (O_j == alpha and max_dist <= 1):
                    kr_norm_upper += tau_j * (self.wasserstein_weight * max_dist - 1)
                    mu_kr_upper += Measure(
                        support=[self.domain[max_dist_index]], coefficients=[tau_j]
                    )
                    if max_dist > 0:
                        number_transported_points_upper += 1
        if kr_norm_lower * self.L > alpha_upper or kr_norm_upper * self.L < alpha_lower:
            # logging.info(
            #     f"alpha-: {alpha_lower}, alpha+: {alpha_upper}, KR-: {kr_norm_lower*self.L}, KR+: {kr_norm_upper*self.L}"
            # )
            return mu_kr_lower, kr_norm_lower, success, number_transported_points_lower
        # logging.info(
        #     f"alpha-: {alpha_lower}, alpha+: {alpha_upper}, KR-: {kr_norm_lower*self.L}, KR+: {kr_norm_upper*self.L}"
        # )

        # There exists a valid solution: construct by convex composition
        success = True
        if kr_norm_lower * self.L == alpha:
            number_transported_points = number_transported_points_lower
        else:
            number_transported_points = number_transported_points_upper
        if kr_norm_lower == kr_norm_upper:
            return mu_kr_lower, kr_norm_lower, success, number_transported_points
        else:
            theta = (alpha / self.L - kr_norm_upper) / (kr_norm_lower - kr_norm_upper)
            mu_kr = mu_kr_lower * theta + mu_kr_upper * (1 - theta)
            return mu_kr, alpha / self.L, success, number_transported_points

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
            )
        else:
            alphas, all_supports_dict = self.compute_alphas_supports(mu, varphi)
            alphas = np.append(alphas, np.inf)
            # logging.info(alphas)
            alpha_intervals = []
            for i, alpha in enumerate(alphas[:-1]):
                alpha_intervals.append((alpha, alpha))
                alpha_intervals.append((alpha, alphas[i + 1]))
            # Loop over the alphas in increasing order to find solution
            # try:
            #     # logging.info(23)
            #     # logging.info(all_supports_dict[23])
            #     # logging.info(25)
            #     logging.info(all_supports_dict[25])
            # except:
            #     pass
            for i, (alpha_lower, alpha_upper) in enumerate(alpha_intervals):
                supports_per_j = {}  # {j: {support indices:, distances:}}
                for j, supports_per_alpha in all_supports_dict.items():
                    incumbent_support_indices = np.array([])
                    incumbent_distances = np.array([])
                    for alpha_, inner_dict in supports_per_alpha.items():
                        # if j == 25:
                        #     logging.info(alpha_)
                        support_indices = inner_dict["support_indices"]
                        support_distances = inner_dict["support_distances"]
                        intermediate_support_indices = inner_dict[
                            "intermediate_support_indices"
                        ]
                        intermediate_support_distances = inner_dict[
                            "intermediate_support_distances"
                        ]
                        if alpha_ == alpha_lower and alpha_ == alpha_upper:
                            # if j == 25:
                            #     logging.info("FFFF" * 100)
                            #     logging.info(alpha_)
                            incumbent_support_indices = support_indices.copy()
                            incumbent_distances = support_distances.copy()
                            break
                        elif alpha_ == alpha_lower:
                            incumbent_support_indices = (
                                intermediate_support_indices.copy()
                            )
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
                # try:
                #     # logging.info(23)
                #     # logging.info(supports_per_j[23])
                #     # logging.info(25)
                #     # logging.info(supports_per_j[25])
                # except:
                #     pass

                if not i:
                    mu_plus, kr_norm, success, number_transported_points = (
                        self.kr_subproblem_first_alpha(
                            alpha_lower, mu, varphi, supports_per_j, log_results
                        )
                    )
                    # if not success:
                    #     logging.info(f"alpha: {alpha_lower}, KR norm: {kr_norm*self.L}")
                else:
                    mu_plus, kr_norm, success, number_transported_points = (
                        self.kr_subproblem(
                            alpha_lower,
                            alpha_upper,
                            mu,
                            varphi,
                            supports_per_j,
                            log_results,
                        )
                    )

                if success:
                    # kr_norm_cvx = self.compute_kr_norm(mu_plus, mu)
                    # logging.info(f"KR norm: {kr_norm*self.L}")
                    # logging.info(f"KR norm cvx: {kr_norm_cvx*self.L}")
                    # if (
                    #     alpha_lower / self.L - 1e-6 <= kr_norm
                    #     and alpha_upper / self.L + 1e-6 >= kr_norm
                    # ):
                    # Check KR descent:
                    diff = self.j(mu_plus) - self.j(mu)
                    kr_rhs = (
                        -mu_plus.duality_pairing(p_mu)
                        + self.beta * np.linalg.norm(mu_plus.coefficients, ord=1)
                        + mu.duality_pairing(p_mu)
                        - self.beta * np.linalg.norm(mu.coefficients, ord=1)
                        + 0.5 * self.L * kr_norm**2
                    )
                    if diff > kr_rhs:
                        logging.warning(f"KR Descent condition failed")
                        descent_condition = False
                    return mu_plus, descent_condition, number_transported_points
                    # else:
                    #     logging.warning("Mismatch in KR norm computation")

        logging.warning("No KR solution found")

    def solve(
        self,
        mu_0: Measure = Measure(),
        max_time: float = 60.0,
        max_iter: int = 250,
        log_results: bool = True,
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

            # mu, descent_condition, number_transported_points = self.kr_step(
            #     mu, p_mu, varphi, log_results
            # )
            self.L = max(self.L * self.L_reduce_factor, self.L_min)
            descent_condition = False
            while not descent_condition:
                mu, descent_condition, number_transported_points = self.kr_step(
                    mu, p_mu, varphi, log_results
                )
                if not descent_condition:
                    self.L = min(self.L * self.L_increase_factor, self.L_max)

            # update metrics
            times.append(time.perf_counter() - initial_time)
            supports.append(len(mu.support))
            objectives.append(self.j(mu))

            if log_results:
                logging.info(
                    f"{k}: L:{self.L:.3E}, transported points: {number_transported_points}, support {supports[-1]}, objective: {objectives[-1]:.12E}"
                )
                # logging.info("=" * 100)
            k += 1
        logging.info(
            f"KR Prox Grad exited after {k} iterations and {times[-1]:.3f}s with final sparsity of {supports[-1]} and objective {objectives[-1]:.12E}"
        )
        return mu, objectives, times, supports
