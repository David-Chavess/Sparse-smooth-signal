from __future__ import annotations

import numpy as np
from pycsou.func import SquaredL2Loss, DiffFuncHStack, NullDifferentiableFunctional, NullProximableFunctional, \
    ProxFuncHStack, L1Norm, SquaredL2Norm
from pycsou.linop import LinOpHStack
from pycsou.opt import APGD

from src.solver import Solver, MyOperator


class SparseSmoothSolver(Solver):

    def __init__(self, y: np.ndarray, operator: np.ndarray, lambda1: float, lambda2: float) -> None:
        super().__init__(y, operator)

        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def solve(self) -> (np.ndarray, np.ndarray):

        H = MyOperator(self.operator)
        H.compute_lipschitz_cst()

        stack = LinOpHStack(H, H, n_jobs=-1)
        stack.compute_lipschitz_cst()

        l22_loss = (1 / 2) * SquaredL2Loss(H.shape[0], self.y)
        F = l22_loss * stack

        L = self.lambda2 * SquaredL2Norm(H.shape[1])
        F = F + DiffFuncHStack(NullDifferentiableFunctional(H.shape[1]), L)

        G = ProxFuncHStack(self.lambda2 * L1Norm(H.shape[1]), NullProximableFunctional(H.shape[1]))

        pds = APGD(2 * self.operator.shape[1], F=F, G=G, verbose=None)
        estimate, converged, diagnostics = pds.iterate()
        x = estimate['iterand']
        x1 = x[:self.operator.shape[1]]
        x2 = x[self.operator.shape[1]:]
        return x1, x2
