from __future__ import annotations

import numpy as np
from pycsou.core import LinearOperator
from pycsou.func import SquaredL2Loss, DiffFuncHStack, NullDifferentiableFunctional, NullProximableFunctional, \
    ProxFuncHStack, L1Norm, SquaredL2Norm
from pycsou.linop import LinOpHStack
from pycsou.opt import APGD

from src.solver import Solver


class SparseSmoothSolver(Solver):
    """
    Solver using APGD algorithm to solve the inverse problem of Hx = y, where H is an operator and x an image,
    with 2 regularization term, L1 and L2.
    """

    def __init__(self, y: np.ndarray, operator: LinearOperator, lambda1: float = 0.1, lambda2: float = 0.1,
                 l2operator: None | LinearOperator = None) -> None:
        """
        Parameters
        ----------
        y: np.ndarray
            Measurements y used in the inverse problem obtained by the linear measurement operator.
        operator: LinearOperator
            Linear operator used for measurements.
        lambda1: float
            Weight of the L1 regularization term.
        lambda2: float
            Weight of the L2 regularization term.
        l2operator: None | LinearOperator
            Operator used in the L2 regularization term if any.
        """
        super().__init__(y, operator)

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.l2operator = l2operator

    def solve(self) -> (np.ndarray, np.ndarray):

        H = self.operator
        if np.isinf(H.lipschitz_cst):
            H.compute_lipschitz_cst()

        stack = LinOpHStack(H, H, n_jobs=-1)

        l22_loss = (1 / 2) * SquaredL2Loss(H.shape[0], self.y)
        F = l22_loss * stack

        if self.lambda2 != 0.0:
            if isinstance(self.l2operator, LinearOperator):
                L = self.lambda2 * SquaredL2Norm(self.l2operator.shape[0]) * self.l2operator
            else:
                L = self.lambda2 * SquaredL2Norm(H.shape[1])

            F = F + DiffFuncHStack(NullDifferentiableFunctional(H.shape[1]), L, n_jobs=-1)

        if self.lambda1 == 0.0:
            G = NullProximableFunctional(2*H.shape[1])
        else:
            G = ProxFuncHStack(self.lambda1 * L1Norm(H.shape[1]), NullProximableFunctional(H.shape[1]), n_jobs=-1)

        apgd = APGD(2 * self.operator.shape[1], F=F, G=G, acceleration='CD', verbose=1)
        estimate, converged, diagnostics = apgd.iterate()
        x = estimate['iterand']
        x1 = x[:self.operator.shape[1]]
        x2 = x[self.operator.shape[1]:]
        return x1, x2
