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
        s = SparseSmoothSignal(self.dim, measurement_operator=-1, psnr=80)
        psnr = 20 * np.log10(np.max(np.abs(s.y0))) - 10 * np.log10(np.var(s.y - s.y0))
        assert np.allclose(psnr, 80, atol=0.1)
        s = SparseSmoothSignal(self.dim, measurement_operator=-1, psnr=0)
        psnr = 20 * np.log10(np.max(np.abs(s.y0))) - 10 * np.log10(np.var(s.y - s.y0))
        assert np.allclose(psnr, 0, atol=0.1)

    def test_cache(self) -> None:
        s = SparseSmoothSignal(self.dim, measurement_operator=-1)
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
        s = SparseSmoothSignal(self.dim, measurement_operator=-1)
        x = np.random.randint(1000, size=self.dim).ravel()
        y = np.random.randint(1000, size=self.dim).ravel()

        free_op = MyMatrixFreeOperator(self.dim)
        assert np.allclose((s.H @ x), free_op(x))
        assert np.allclose((s.H.adjoint(y)), free_op.adjoint(y))

        s.random_measurement_operator(self.dim[0])
        free_op = MyMatrixFreeOperator(self.dim, s.random_lines)
        assert np.allclose((s.H @ x), free_op(x))
        y = y[s.random_lines]
        assert np.allclose((s.H.adjoint(y)), free_op.adjoint(y))

        free_op = MyMatrixFreeOperator(self.dim)
        y = np.random.random((self.dim[0]*self.dim[1], 2)).view(np.complex128)
        assert np.allclose(free_op.pinv(y), free_op.adjoint(y))
        assert np.allclose(free_op.adjoint(y), free_op.transpose(np.conj(y)).real)


if __name__ == '__main__':
    test = Test()
    test.test_operator()
    test.test_noise()
    test.test_cache()
    test.test_MyMatrixFreeOperator()
    print("Test Done")

