from __future__ import annotations

from typing import List

from pycsou.core import LinearOperator

from src.sparse_smooth_signal import SparseSmoothSignal
from src.sparse_smooth_solver import SparseSmoothSolver


def test_solvers(s: SparseSmoothSignal, lambda1: float, lambda2: float):
    sol = SparseSmoothSolver(s.y, s.H, lambda1, lambda2, "deriv1")
    x_ss = sol.solve()

    sol = SparseSmoothSolver(s.y, s.H, 0.0, lambda2, "deriv1")
    x_tik = sol.solve()

    sol = SparseSmoothSolver(s.y, s.H, lambda1, 0.0)
    x_lasso = sol.solve()

    s1 = SparseSmoothSignal(s.dim, sparse=x_ss[0].reshape(dim), smooth=x_ss[1].reshape(dim), measurement_operator=s.H)
    s2 = SparseSmoothSignal(s.dim, sparse=x_tik[0].reshape(dim), smooth=x_tik[1].reshape(dim), measurement_operator=s.H)
    s3 = SparseSmoothSignal(s.dim, sparse=x_lasso[0].reshape(dim), smooth=x_lasso[1].reshape(dim),
                            measurement_operator=s.H)

    s1.plot("Sparse + Smooth")
    s2.plot("Tik")
    s3.plot("Lasso")
    s.plot("Base")
    s.show()


def test(s: SparseSmoothSignal, H: List[float], lambda1: List[float], lambda2: List[float],
         operators: List[None | str | LinearOperator], psnr: List[float]):
    size = s.dim[0] * s.dim[1]
    for h in H:
        s.random_measurement_operator(int(h * size))
        for p in psnr:
            s.gaussian_noise(p)
            for l1 in lambda1:
                for l2 in lambda2:
                    for op in operators:
                        solver = SparseSmoothSolver(s.y, s.H, l1, l2, op)
                        x1, x2 = solver.solve()
                        s1 = SparseSmoothSignal(s.dim, sparse=x1.reshape(s.dim), smooth=x2.reshape(s.dim),
                                                measurement_operator=s.H)
                        s1.plot(f"Lambda1:{l1}, Lambda2:{l2}, H:{h * 100}% lines, psnr:{p}, l2 operator:{op}")

    s.plot("Base")
    s.show()


if __name__ == '__main__':
    dim = (16, 16)
    s = SparseSmoothSignal(dim)
    test(s, [0.1, 0.2, 0.3, 0.4, 0.5], [0.1], [0.1], [None, "deriv1"], [50.0])
