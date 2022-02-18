from __future__ import annotations

import numpy as np
from scipy.linalg import dft
from scipy.sparse import random as sparse_random
import matplotlib.pyplot as plt


class SpareSmoothSignal:
    """
    Base class for the spare and smooth signal.

    The signal is composed of 2 signals, one sparse and one smooth, x = x_sparse + x_smooth
    yo is the prefect signal obtained through a linear measurement operator H of the signal such that y0 = H @ x
    y is the signal yo with some error represented by a gaussian white noise
    """

    def __init__(self, dim: (int, int), sparse: None | np.ndarray = None, smooth: None | np.ndarray = None,
                 measurement_operator: None | np.ndarray = None):
        assert dim[0] >= 0 and dim[1] >= 0, "Negative dimension is not valid"

        self.__dim = dim
        self.__size = dim[0] * dim[1]
        self.__y_size = self.__size

        if sparse is not None:
            assert sparse.shape == dim, "Sparse is not the same shape as dim"
            self.__sparse = sparse
        else:
            self.random_sparse()

        if smooth is not None:
            assert smooth.shape == dim, "Smooth is not the same size as dim"
            self.__smooth = smooth
        else:
            self.random_smooth()

        if measurement_operator is np.ndarray:
            assert measurement_operator.shape[1] == self.__size, "Measurement operator shape does not match dim"
            self.__measurement_operator = measurement_operator
        else:
            self.__measurement_operator = None

        self.__noise = None

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
    def sigma(self) -> None | np.float_:
        return self.__sigma

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

    def random_sparse(self) -> None:
        self.__sparse = np.ravel(sparse_random(self.__dim[0], self.__dim[1]).toarray())

    def random_smooth(self) -> None:
        self.__smooth = np.random.rand(self.__dim[0], self.__dim[1]).ravel()

    def random_measurement_operator(self, size: int) -> None:
        assert self.__size >= size >= 0
        self.__y_size = size
        dft_mtx = dft(self.__size)
        rand = np.random.choice(self.__size, size, replace=False)
        self.__measurement_operator = dft_mtx[rand]
        self.__noise = None

    def gaussian_noise(self, sigma=1) -> None:
        self.__noise = np.random.normal(0, sigma, self.__y_size)

