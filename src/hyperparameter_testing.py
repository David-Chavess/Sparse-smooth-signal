from pycsou.linop import IdentityOperator

from src.sparse_smooth_solver import SparseSmoothSolver
from src.sparse_smooth_signal import SparseSmoothSignal
import numpy as np

dim = (16, 16)


def test():
    size = dim[0] * dim[1]
    s = SparseSmoothSignal(dim)
    s.plot("Base")

    sol = SparseSmoothSolver(s.y, s.H, 0.1, 0.1, "deriv1")
    x1, x2 = sol.solve()
    print(s.H.shape)
    print(s.y.shape)
    SparseSmoothSignal(dim, x1.reshape(dim), x2.reshape(dim), s.H).plot("Sparse + Smooth : [0.1, 0.1, D, 100%]")

    s.random_measurement_operator(int(0.9*size))
    print(s.H.shape)
    print(s.y.shape)
    sol = SparseSmoothSolver(s.y, s.H, 0.1, 0.1, "deriv1")
    x1, x2 = sol.solve()
    SparseSmoothSignal(dim, x1.reshape(dim), x2.reshape(dim), s.H).plot("Sparse + Smooth : [0.1, 0.1, D, 90%]")

    s.random_measurement_operator(int(0.75 * size))
    sol = SparseSmoothSolver(s.y, s.H, 0.1, 0.1, "deriv1")
    x1, x2 = sol.solve()
    SparseSmoothSignal(dim, x1.reshape(dim), x2.reshape(dim), s.H).plot("Sparse + Smooth : [0.1, 0.1, D, 75%]")

    s.random_measurement_operator(int(0.5 * size))
    sol = SparseSmoothSolver(s.y, s.H, 0.1, 0.1, "deriv1")
    x1, x2 = sol.solve()
    SparseSmoothSignal(dim, x1.reshape(dim), x2.reshape(dim), s.H).plot("Sparse + Smooth : [0.1, 0.1, D, 50%]")

    s.random_measurement_operator(int(0.25 * size))
    sol = SparseSmoothSolver(s.y, s.H, 0.1, 0.1, "deriv1")
    x1, x2 = sol.solve()
    SparseSmoothSignal(dim, x1.reshape(dim), x2.reshape(dim), s.H).plot("Sparse + Smooth : [0.1, 0.1, D, 25%]")

    s.random_measurement_operator(int(0.1 * size))
    sol = SparseSmoothSolver(s.y, s.H, 0.1, 0.1, "deriv1")
    x1, x2 = sol.solve()
    SparseSmoothSignal(dim, x1.reshape(dim), x2.reshape(dim), s.H).plot("Sparse + Smooth : [0.1, 0.1, D, 10%]")

    s.show()


if __name__ == '__main__':
    test()
