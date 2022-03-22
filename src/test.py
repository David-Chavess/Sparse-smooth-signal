import timeit

import numpy as np

from src.solver import MyMatrixFreeOperator
from src.sparse_smooth_signal import SparseSmoothSignal
from src.sparse_smooth_solver import SparseSmoothSolver


class Test:

    def __init__(self):
        self.dim = (100, 100)

    def test_operator(self) -> None:
        operator = SparseSmoothSignal.create_measurement_operator(self.dim)
        vec = np.random.randint(1000, size=self.dim)
        assert np.allclose((operator @ vec.ravel()).reshape(self.dim), np.fft.fft2(vec, norm='ortho'))

    def test_noise(self) -> None:
        s = SparseSmoothSignal(self.dim, psnr=80)
        psnr = 20 * np.log10(np.max(np.abs(s.y0))) - 10 * np.log10(np.var(s.y - s.y0))
        assert np.allclose(psnr, 80, atol=0.1)
        s = SparseSmoothSignal(self.dim, psnr=0)
        psnr = 20 * np.log10(np.max(np.abs(s.y0))) - 10 * np.log10(np.var(s.y - s.y0))
        assert np.allclose(psnr, 0, atol=0.1)

    def test_cache(self) -> None:
        s = SparseSmoothSignal(self.dim)
        smooth = s.smooth
        s.random_smooth()
        assert not np.allclose(s.smooth, smooth)

        sparse = s.sparse
        s.random_sparse()
        assert not np.allclose(s.sparse, sparse)

        x = s.x
        y = s.y
        s.random_smooth()
        assert not np.allclose(s.x, x)
        assert not np.allclose(s.y, y)

        x = s.x
        y = s.y
        s.random_sparse()
        assert not np.allclose(s.x, x)
        assert not np.allclose(s.y, y)

    def test_MyMatrixFreeOperator(self) -> None:
        operator = SparseSmoothSignal.create_measurement_operator(self.dim)
        free_op = MyMatrixFreeOperator(self.dim)
        vec = np.random.randint(1000, size=self.dim)
        assert np.allclose((operator @ vec.ravel()), free_op(vec.ravel()))

        s = SparseSmoothSignal(self.dim, measurement_operator=self.dim[0])
        free_op = MyMatrixFreeOperator(self.dim, s.operator_random)
        vec = np.random.randint(1000, size=self.dim)
        assert np.allclose((operator @ vec.ravel()), free_op(vec.ravel()))


if __name__ == '__main__':
    # test = Test()
    # test.test_operator()
    # test.test_noise()
    # test.test_cache()
    # test.test_MyMatrixFreeOperator()
    # print("Test Done")

    dim = (1024, 1024)
    x = 125

    op = MyMatrixFreeOperator(dim)
    s = SparseSmoothSignal(dim, measurement_operator=op)
    s.plot(f"1024x1024, {x}")

    dim = (256, 256)

    op = MyMatrixFreeOperator(dim)
    s = SparseSmoothSignal(dim, measurement_operator=op)
    s.plot(f"256x256, {x}")
    s.show()

    s1 = SparseSmoothSignal(dim)
    s1.plot("Base")

    t1 = timeit.timeit()
    solver = SparseSmoothSolver(s1.y, s1.H, 0.1, 0.1, "deriv1")
    x1, x2 = solver.solve()
    t2 = timeit.timeit()
    s2 = SparseSmoothSignal(dim, sparse=x1.reshape(dim), smooth=x2.reshape(dim), measurement_operator=s1.H)
    s2.plot("op1")

    op = MyMatrixFreeOperator(dim)
    s1.measurement_operator = op

    t3 = timeit.timeit()
    solver = SparseSmoothSolver(s1.y, s1.H, 0.1, 0.1, "deriv1")
    x1, x2 = solver.solve()
    t4 = timeit.timeit()
    s2 = SparseSmoothSignal(dim, sparse=x1.reshape(dim), smooth=x2.reshape(dim), measurement_operator=s1.H)
    s2.plot("op2")

    print(t2-t1)
    print(t4-t3)
    s2.show()

