from __future__ import annotations

import numpy as np
from pycsou.func import SquaredL2Loss, L2Norm, QuadraticForm
from pycsou.linop import DenseLinearOperator
from pycsou.opt import APGD, PrimalDualSplitting

from src.solver import Solver, MyOperator


class TikhonovSolver(Solver):

    def __init__(self, y: np.ndarray, operator: np.ndarray, tikhonov_matrix: None | float | np.ndarray,
                 generalized: bool = False) -> None:
        super().__init__(y, operator)
        if type(tikhonov_matrix) is float:
            self.tikhonov_matrix = tikhonov_matrix * np.eye(operator.shape[1])
        else:
            self.tikhonov_matrix = tikhonov_matrix
        self.generalized = generalized

    def solve(self) -> (np.ndarray, np.ndarray):

        H = MyOperator(self.operator)
        H.compute_lipschitz_cst()

        l22_loss = (1 / 2) * SquaredL2Loss(H.shape[0], self.y)
        F = l22_loss * H

        D = DenseLinearOperator(self.tikhonov_matrix)
        D.compute_lipschitz_cst()

        if self.generalized:
            L = QuadraticForm(H.shape[1], D)
        else:
            L = L2Norm(H.shape[1])

        apgd = PrimalDualSplitting(self.operator.shape[1], F=F, H=L, K=D, verbose=None)
        estimate, converged, diagnostics = apgd.iterate()
        x = estimate['primal_variable']#['iterand']
        return x, x
