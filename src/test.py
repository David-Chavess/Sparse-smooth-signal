from src.sparse_smooth_signal import SparseSmoothSignal
import numpy as np


class Test:

    def test_operator(self) -> None:
        dim = (100, 100)
        operator = SparseSmoothSignal.create_measurement_operator(dim)
        vec = np.random.randint(1000, size=dim)
        assert np.allclose((operator @ vec.ravel()).reshape(dim), np.fft.fft2(vec))


if __name__ == '__main__':
    test = Test()
    test.test_operator()
