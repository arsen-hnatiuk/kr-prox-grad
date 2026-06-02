# An implementation of the semismooth Newton method following Section 3 of https://mediatum.ub.tum.de/doc/1241413/1241413.pdf

import numpy as np
import logging
from typing import Callable

logging.basicConfig(
    level=logging.DEBUG,
)


class SSN:
    def __init__(
        self,
        K: np.ndarray,
        alpha: float,
        target: np.ndarray,
        M: float,
        g: Callable,
        f: Callable,
        grad_f: Callable,
        hess_f: Callable,
        maximum_iterations: int = 1000,
        log_results: bool = True,
    ) -> None:
        self.K = K
        if all(self.K.shape):
            self.machine_precision = 1e-12
            self.target = target
            self.alpha = alpha
            self.g = g
            self.f = f
            self.grad_f = grad_f
            self.p = lambda u: -np.array(self.K.T @ self.grad_f(self.K @ u))  # -f'
            if np.linalg.norm(
                hess_f(np.ones(len(self.target)))
                - hess_f(0.5 * np.ones(len(self.target)))
            ):
                self.hess_f = hess_f
                self.hessian_matrix = None
                self.hessian = lambda u: np.array(
                    self.K.T @ self.hess_f(self.K @ u) @ self.K
                )
            else:
                # Constant Hessian
                self.hess_f = hess_f(np.ones(len(self.target)))
                self.hessian_matrix = self.K.T @ self.hess_f @ self.K
                self.hessian = lambda u: self.hessian_matrix
                self.hess_f = None
            self.j = lambda u: float(self.f(self.K @ u) + self.g(u))
            self.M = M
            self.maximum_iterations = maximum_iterations
            self.log_results = log_results

    def Psi(self, u: np.ndarray) -> np.ndarray:
        # sup_v <p(u),v-u>+g(u)-g(v)
        u = u.copy()
        p = self.p(u)
        constant_part = -np.matmul(p, u) + self.g(u)
        variable_part = max(0, self.M * (np.max(np.absolute(p)) - self.alpha))
        return constant_part + variable_part

    def prox(self, q: np.ndarray) -> np.ndarray:
        q = q.copy()
        to_return = np.zeros(q.shape)
        for i, val in enumerate(q):
            if np.abs(val) > self.alpha:
                to_return[i] = val - self.alpha * np.sign(val)
        return to_return

    def grad_prox(self, q: np.ndarray) -> np.ndarray:
        q = q.copy()
        return np.diag(np.where(np.abs(q) > self.alpha, 1, 0))

    def solve(self, tol: float, u_0: np.ndarray) -> np.ndarray:
        # Semismooth Newton method (globalized via line search)
        if not all(self.K.shape):
            if self.log_results:
                logging.debug("Empty input space, retuning u_0")
            return u_0
        theta = tol  # Set initial value for the step length parameter
        Id = np.identity(len(u_0))
        initial_j = self.j(u_0)
        q = u_0
        prox_q = self.prox(q)  # The actual iterate
        psi_val = min(self.Psi(prox_q), self.Psi(q))
        k = 0
        while psi_val > tol:
            if k > self.maximum_iterations:
                if self.log_results:
                    logging.info(
                        f"SSN in {len(prox_q)} dimensions and tolerance {tol:.3E}: MAX ITERATIONS REACHED, {psi_val:.3E} achieved"
                    )
                if self.j(prox_q) <= initial_j:
                    return prox_q
                else:
                    return u_0
            right_hand = q - prox_q - self.p(prox_q)
            left_hand = Id + (self.hessian(prox_q) - Id) @ self.grad_prox(q)
            theta = theta / 10
            qdiff = tol + 1
            while qdiff >= tol:
                theta = 2 * theta
                try:
                    direction = np.linalg.solve(left_hand + theta * Id, right_hand)
                except np.linalg.LinAlgError:
                    if self.log_results:
                        logging.info(
                            f"SSN in {len(prox_q)} dimensions and tolerance {tol:.3E}: LINEAR SYSTEM NOT SOLVABLE, {psi_val:.3E} achieved"
                        )
                    if self.j(prox_q) <= initial_j:
                        return prox_q
                    else:
                        return u_0
                qnew = q - direction
                prox_qnew = self.prox(qnew)
                qdiff = self.j(prox_qnew) - self.j(prox_q)
            q = qnew
            prox_q = prox_qnew
            self.M = float(min(self.M, self.j(prox_q) / self.alpha))
            psi_val = self.Psi(prox_q)
            k += 1

        # if self.log_results:
        #     logging.info(
        #         f"SSN in {len(prox_q)} dimensions converged in {k} iterations to tolerance {tol:.3E}"
        #     )
        if self.j(prox_q) <= initial_j:
            return prox_q
        else:
            return u_0


# if __name__ == "__main__":
#     K = np.array([[-1, 2, 0], [3, 0, 0], [-1, -2, -1]])
#     u = np.array([-1, -1, -1])
#     y = np.array([1, 0, 4])
#     sn = SSN(K, 1, y, 20)
#     print(sn.solve(1e-12, u))
