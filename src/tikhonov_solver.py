from __future__ import annotations

import numpy as np
from pycsou.func import SquaredL2Loss, QuadraticForm, DiffFuncHStack, NullDifferentiableFunctional
from pycsou.linop import DenseLinearOperator, IdentityOperator, LinOpHStack
from pycsou.opt import APGD

from src.solver import Solver, MyOperator


class TikhonovSolver(Solver):

    def __init__(self, y: np.ndarray, operator: np.ndarray, tikhonov_matrix: None | float | np.ndarray,
                 generalized: bool = False) -> None:
        super().__init__(y, operator)

        self.tikhonov_matrix = tikhonov_matrix
        self.generalized = generalized

    def solve(self) -> (np.ndarray, np.ndarray):

        H = MyOperator(self.operator)
        H.compute_lipschitz_cst()

        stack = LinOpHStack(H, H, n_jobs=-1)
        stack.compute_lipschitz_cst()

        l22_loss = (1 / 2) * SquaredL2Loss(H.shape[0], self.y)
        F = l22_loss * stack

        if self.tikhonov_matrix is not None:
            if self.generalized:
                L = QuadraticForm(H.shape[1], DenseLinearOperator(self.tikhonov_matrix))
            else:
                if type(self.tikhonov_matrix) is float:
                    L = QuadraticForm(H.shape[1], self.tikhonov_matrix * IdentityOperator(H.shape[1]))
                else:
                    L = QuadraticForm(H.shape[1], DenseLinearOperator(self.tikhonov_matrix.T @ self.tikhonov_matrix))

            F = F + DiffFuncHStack(NullDifferentiableFunctional(H.shape[1]), L)

        pds = APGD(2 * self.operator.shape[1], F=F, verbose=None)
        estimate, converged, diagnostics = pds.iterate()
        x = estimate['iterand']
        x1 = x[:self.operator.shape[1]]
        x2 = x[self.operator.shape[1]:]
        return x1, x2
