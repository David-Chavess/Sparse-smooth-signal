from __future__ import annotations

import numpy as np
from pycsou.core import LinearOperator
from pycsou.func import SquaredL2Loss, SquaredL2Norm
from pycsou.linop import IdentityOperator
from pycsou.opt import APGD

from src.solver import Solver


class TikhonovSolver(Solver):

    def __init__(self, y: np.ndarray, operator: LinearOperator, tikhonov_matrix: None | float | LinearOperator) -> None:
        super().__init__(y, operator)
        self.tikhonov_matrix = tikhonov_matrix

    def solve(self) -> (np.ndarray, np.ndarray):

        H = self.operator
        H.compute_lipschitz_cst()

        l22_loss = (1 / 2) * SquaredL2Loss(H.shape[0], self.y)
        F = l22_loss * H

        if self.tikhonov_matrix is not None:
            if type(self.tikhonov_matrix) is float:
                L = SquaredL2Norm(H.shape[1]) * (self.tikhonov_matrix * IdentityOperator(H.shape[1]))
            else:
                L = SquaredL2Norm(H.shape[1]) * self.tikhonov_matrix
            F = F + L

        pds = APGD(self.operator.shape[1], F=F, verbose=None)
        estimate, converged, diagnostics = pds.iterate()
        x = estimate['iterand']
        return None, x
