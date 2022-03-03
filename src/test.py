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
    s = SparseSmoothSignal(dim)
    s.plot()
    s.show()
