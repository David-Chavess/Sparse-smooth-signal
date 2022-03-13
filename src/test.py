from pycsou.linop import FirstDerivative

from src.lasso_solver import LassoSolver
from src.tikhonov_solver import TikhonovSolver
from src.sparse_smooth_signal import SparseSmoothSignal
import numpy as np


class Test:

    def __init__(self):
        self.dim = (100, 100)

    def test_operator(self) -> None:
        operator = SparseSmoothSignal.create_measurement_operator(self.dim)
        vec = np.random.randint(1000, size=self.dim)
        assert np.allclose((operator @ vec.ravel()).reshape(self.dim), np.fft.fft2(vec))

    def test_noise(self) -> None:
        s = SparseSmoothSignal(self.dim)
        s.gaussian_noise(80)
        psnr = 20 * np.log10(np.max(np.abs(s.y0))) - 10 * np.log10(np.var(s.y - s.y0))
        assert np.allclose(psnr, 80, rtol=0.1)

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


if __name__ == '__main__':
    #test = Test()
    #test.test_operator()
    #test.test_noise()
    #test.test_cache()
    #print("Done")

    dim = (100, 100)
    s = SparseSmoothSignal(dim, measurement_operator=5000)

    sol = TikhonovSolver(s.y, s.measurement_operator, 0.1)
    x, _ = sol.solve()

    s.plot("Base")
    s.show()
    SparseSmoothSignal(dim, smooth=x.reshape(dim), sparse=np.zeros(dim), measurement_operator=s.H).plot()

    D = FirstDerivative(10000)
    D.compute_lipschitz_cst()
    sol = LassoSolver(s.y, s.measurement_operator, 0.1, D)
    x, _ = sol.solve()

    SparseSmoothSignal(dim, smooth=np.zeros(dim), sparse=x.reshape(dim), measurement_operator=s.H).plot()
    s.show()

