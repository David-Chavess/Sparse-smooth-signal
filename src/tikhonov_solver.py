from __future__ import annotations

import numpy as np
from pycsou.core import LinearOperator
from pycsou.func import SquaredL2Loss, SquaredL2Norm
from pycsou.opt import APGD

from src.solver import Solver


class TikhonovSolver(Solver):
    """
    Solver solving the Tikhonov regularization problem
    """

    def __init__(self, y: np.ndarray, operator: LinearOperator, lambda_: float,
                 l2_op: None | LinearOperator = None) -> None:
        """
        Parameters
        ----------
        y: np.ndarray
            Measurements y used in the inverse problem obtained by the linear measurement operator.
        operator: LinearOperator
            Linear operator used for measurements.
        lambda_: float
            Weight of the L2 regularization term.
        l2_op: None | LinearOperator
            Operator used in the L2 regularization term if any.
        """
        super().__init__(y, operator)
        self.l2_op = l2_op
        self.lambda_ = lambda_

    def solve(self) -> (np.ndarray, np.ndarray):

        H = self.operator
        H.compute_lipschitz_cst()

        l22_loss = (1 / 2) * SquaredL2Loss(H.shape[0], self.y)
        F = l22_loss * H

        L = self.lambda_ * SquaredL2Norm(H.shape[1])

        if self.l2_op is not None:
            L = SquaredL2Norm(H.shape[1]) * self.l2_op

        F = F + L

        pds = APGD(self.operator.shape[1], F=F, verbose=None)
        estimate, converged, diagnostics = pds.iterate()
        x = estimate['iterand']
        return None, x
