from __future__ import annotations

import numpy as np
from pycsou.core import LinearOperator
from pycsou.func import SquaredL2Loss, DiffFuncHStack, NullDifferentiableFunctional, NullProximableFunctional, \
    ProxFuncHStack, L1Norm, SquaredL2Norm
from pycsou.linop import LinOpHStack, FirstDerivative
from pycsou.opt import APGD

from src.solver import Solver


class SparseSmoothSolver(Solver):

    def __init__(self, y: np.ndarray, operator: LinearOperator, lambda1: float = 0.1, lambda2: float = 0.1,
                 l2operator: None | str | LinearOperator = None) -> None:
        super().__init__(y, operator)

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        if isinstance(l2operator, str):
            if l2operator == "deriv1":
                l2operator = FirstDerivative(operator.shape[1])
                l2operator.compute_lipschitz_cst()

        self.l2operator = l2operator

    def solve(self) -> (np.ndarray, np.ndarray):

        H = self.operator
        H.compute_lipschitz_cst()

        stack = LinOpHStack(H, H, n_jobs=-1)
        stack.compute_lipschitz_cst()

        l22_loss = (1 / 2) * SquaredL2Loss(H.shape[0], self.y)
        F = l22_loss * stack

        if self.lambda2 != 0.0:
            L = self.lambda2 * SquaredL2Norm(H.shape[1])

            if isinstance(self.l2operator, LinearOperator):
                L = L * self.l2operator

            F = F + DiffFuncHStack(NullDifferentiableFunctional(H.shape[1]), L, n_jobs=-1)

        if self.lambda1 == 0.0:
            G = NullProximableFunctional(2*H.shape[1])
        else:
            G = ProxFuncHStack(self.lambda1 * L1Norm(H.shape[1]), NullProximableFunctional(H.shape[1]), n_jobs=-1)

        apgd = APGD(2 * self.operator.shape[1], F=F, G=G, acceleration='CD', max_iter=200, verbose=1)
        estimate, converged, diagnostics = apgd.iterate()
        x = estimate['iterand']
        x1 = x[:self.operator.shape[1]]
        x2 = x[self.operator.shape[1]:]
        return x1, x2
