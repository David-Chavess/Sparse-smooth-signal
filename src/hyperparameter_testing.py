from __future__ import annotations

from typing import List

import numpy as np
from pycsou.core import LinearOperator

from src.lasso_solver import LassoSolver
from src.solver import MyMatrixFreeOperator
from src.sparse_smooth_signal import SparseSmoothSignal
from src.sparse_smooth_solver import SparseSmoothSolver
from src.tikhonov_solver import TikhonovSolver


def loss(x1, x2):
    return np.sum((x1 - x2) ** 2)


def test_solvers(s: SparseSmoothSignal, lambda1: (float, float), lambda2: (float, float | LinearOperator)):
    sol = SparseSmoothSolver(s.y, s.H, lambda1[0], lambda2[0], "deriv1")
    x_ss = sol.solve()

    sol = TikhonovSolver(s.y, s.H, lambda2[1])
    x_tik = sol.solve()

    sol = LassoSolver(s.y, s.H, lambda1[1])
    x_lasso = sol.solve()

    s1 = SparseSmoothSignal(s.dim, sparse=x_ss[0].reshape(dim), smooth=x_ss[1].reshape(dim), measurement_operator=s.H)
    s2 = SparseSmoothSignal(s.dim, sparse=np.zeros(dim), smooth=x_tik[1].reshape(dim), measurement_operator=s.H)
    s3 = SparseSmoothSignal(s.dim, sparse=x_lasso[0].reshape(dim), smooth=np.zeros(dim), measurement_operator=s.H)

    s1.plot("Sparse + Smooth")
    s2.plot("Tik")
    s3.plot("Lasso")
    s.plot("Base")

    print(f"Sparse + Smooth loss : {loss(s.x, s1.x)}")
    print(f"Tik loss : {loss(s.x, s2.x)}")
    print(f"Lasso loss : {loss(s.x, s3.x)}")
    s.show()


def test(s: SparseSmoothSignal, H: List[float], lambda1: List[float], lambda2: List[float],
         operators: List[None | str | LinearOperator], psnr: List[float]):
    size = s.dim[0] * s.dim[1]
    loss_x = {}
    loss_x1 = {}
    loss_x2 = {}
    for h in H:
        s.random_measurement_operator(int(h * size))
        op = MyMatrixFreeOperator(s.dim, s.random_lines)
        s.H = op
        for p in psnr:
            s.gaussian_noise(p)
            for l1 in lambda1:
                for l2 in lambda2:
                    for op in operators:
                        solver = SparseSmoothSolver(s.y, s.H, l1, l2, op)
                        x1, x2 = solver.solve()
                        s1 = SparseSmoothSignal(s.dim, sparse=x1.reshape(s.dim), smooth=x2.reshape(s.dim),
                                                measurement_operator=s.H)
                        name = f"Lambda1:{l1}, Lambda2:{l2}, H:{h * 100}% lines, psnr:{p}, l2 operator:{op}"
                        s1.plot(name)
                        loss_x[name] = np.sum((s.x - s1.x) ** 2)
                        loss_x1[name] = np.sum((s.sparse - s1.sparse) ** 2)
                        loss_x2[name] = np.sum((s.smooth - s1.smooth) ** 2)

    print("Sparse + Smooth:")
    for x in sorted(loss_x, key=loss_x.get):
        print(f"{x} : {loss_x[x]}")
    print("Sparse:")
    for x in sorted(loss_x1, key=loss_x1.get):
        print(f"{x} : {loss_x1[x]}")
    print("Smooth:")
    for x in sorted(loss_x2, key=loss_x2.get):
        print(f"{x} : {loss_x2[x]}")

    s.plot("Base")
    s.show()


if __name__ == '__main__':
    dim = (32, 32)
    op = MyMatrixFreeOperator(dim)
    s1 = SparseSmoothSignal(dim, measurement_operator=op)
    test_solvers(s1, (0.1, 0.1), (0.1, 0.1))
    # test(s1, [0.25, 0.5], [0.1], [0.1], [None, "deriv1"], [20.0, 50.0])
