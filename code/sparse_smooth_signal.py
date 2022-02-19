from __future__ import annotations

import numpy as np
from scipy.linalg import dft
from scipy import sparse
import matplotlib.pyplot as plt


class SpareSmoothSignal:
    """
    Base class for the spare and smooth signal.

    The signal is composed of 2 signals, one sparse and one smooth, x = x_sparse + x_smooth
    yo is the prefect signal obtained through a linear measurement operator H of the signal such that y0 = H @ x
    y is the signal yo with some error represented by a gaussian white noise
    """

    def __init__(self, dim: (int, int), sparse: None | np.ndarray = None, smooth: None | np.ndarray = None,
                 measurement_operator: None | np.ndarray = None, variance: float = 1):
        assert dim[0] >= 0 and dim[1] >= 0, "Negative dimension is not valid"

        self.__dim = dim
        self.__size = dim[0] * dim[1]
        self.__y_size = self.__size

        if sparse is not None:
            assert sparse.shape == dim, "Sparse is not the same shape as dim"
            self.__sparse = sparse.ravel()
        else:
            self.random_sparse()

        if smooth is not None:
            assert smooth.shape == dim, "Smooth is not the same size as dim"
            self.__smooth = smooth.ravel()
        else:
            self.random_smooth()

        if measurement_operator is np.ndarray:
            assert measurement_operator.shape[1] == self.__size, "Measurement operator shape does not match dim"
            self.__measurement_operator = measurement_operator
        else:
            self.random_measurement_operator(self.__y_size)
        self.__variance = variance
        self.__noise = None
        self.gaussian_noise()

    @property
    def dim(self) -> (int, int):
        return self.__dim

    @property
    def sparse(self) -> None | np.ndarray:
        return self.__sparse

    @property
    def smooth(self) -> None | np.ndarray:
        return self.__smooth

    @property
    def measurement_operator(self) -> None | np.ndarray:
        return self.__measurement_operator

    @property
    def x(self) -> None | np.ndarray:
        return self.__sparse + self.__smooth

    @property
    def H(self) -> None | np.ndarray:
        return self.measurement_operator

    @property
    def y0(self) -> None | np.ndarray:
        if self.__measurement_operator is not None:
            return self.H @ self.x
        return None

    @property
    def y(self) -> None | np.ndarray:
        if self.__noise is not None:
            return self.y0 + self.__noise
        return None

    @property
    def noise(self) -> np.ndarray:
        return self.__noise

    def random_sparse(self) -> None:
        self.__sparse = sparse.random(self.__dim[0], self.__dim[1], density=0.25, data_rvs=np.random.randn)\
            .toarray().ravel()

    def random_smooth(self) -> None:
        amp = np.random.randint(0, 5, 2)
        freq = 2*np.pi*np.random.rand(2)
        t = np.linspace(0, self.__size/10, self.__size)
        self.__smooth = amp[0]*np.sin(t/freq[0]) + amp[1]*np.cos(t/freq[1])

    def random_measurement_operator(self, size: int) -> None:
        assert self.__size >= size >= 0
        self.__y_size = size
        dft_mtx = dft(self.__size)
        rand = np.random.choice(self.__size, size, replace=False)
        self.__measurement_operator = dft_mtx[rand]
        self.__noise = None

    def gaussian_noise(self, variance: float = None) -> None:
        if variance is None:
            variance = self.__variance
        else:
            self.__variance = variance
        self.__noise = np.random.normal(0, variance, self.__y_size)

    def draw_sparse(self) -> None:
        plt.plot(self.sparse, label="sparse")

    def draw_smooth(self) -> None:
        plt.plot(self.smooth, label="smooth")

    def draw_x(self) -> None:
        plt.plot(self.x, label="x")

    def draw_y0(self) -> None:
        plt.plot(self.y0, label="y0")

    def draw_y(self) -> None:
        plt.plot(self.y, label="y")

    def draw_noise(self) -> None:
        plt.plot(self.noise, label="noise")

    def show(self) -> None:
        plt.title("Spare + Smooth Signal")
        plt.ylabel("Amplitude")
        plt.legend(loc="best")
        plt.show()
