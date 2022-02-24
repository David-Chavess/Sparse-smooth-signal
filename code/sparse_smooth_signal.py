from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.linalg import dft
import scipy.sparse as sp
import matplotlib.pyplot as plt


class SparseSmoothSignal:
    """
    Base class for the sparse and smooth signal.

    The signal is composed of 2 signals, one sparse and one smooth, x = x_sparse + x_smooth
    yo is the prefect signal obtained through a linear measurement operator H of the signal such that y0 = H @ x
    y is the signal yo with some error represented by a gaussian white noise

    Attributes
    ----------

    dim : Tuple[int ,int]
        shape of the signal x
    sparse : np.ndarray
        matrix representing the sparse part of the signal
    smooth : np.ndarray
        matrix representing the smooth part of the signal
    x : np.ndarray
        signal x sum of sparse and smooth
    measurement_operator : np.ndarray 
        matrix representing the linear sensing measurement operator used
    H : np.ndarray 
        alias for measurement_operator
    variance : np.float64
        variance of the gaussian white noise added 
    noise :
        gaussian white noise added
    yo : np.ndarray
        prefect output signal
    y : np.ndarray
        output signal

    Methods
    -------
    random_sparse() -> None
        Creates a new random sparse component
    random_smooth() -> None
        Creates a new random smooth component
    random_measurement_operator(size: int) -> None
        Creates a new random measurement operator with size random lines of the DFT matrix
    gaussian_noise(variance: np.float64 = None) -> None:
        Creates a new gaussian white noise
    plot() -> None
        Plot all signals in 2d
    """

    def __init__(self, dim: Tuple[int, int], sparse: None | np.ndarray = None, smooth: None | np.ndarray = None,
                 measurement_operator: None | np.ndarray = None, variance: np.float64 = 1):
        """
        Parameters
        ----------
        dim : Tuple[int ,int]
            shape of the signal x
        sparse : None | np.ndarray
            matrix representing the sparse part of the signal
        smooth : None | np.ndarray
            matrix representing the smooth part of the signal
        measurement_operator : None | np.ndarray 
            matrix representing the linear sensing measurement operator used
        variance : np.float64
            variance of the gaussian white noise added 

        For any optionnal argumant if not specified the corresponding value will be random 
        except variance which is 1 by default
        """
        assert dim[0] >= 0 and dim[1] >= 0, "Negative dimension is not valid"

        self.__dim = dim
        # length of the signal x
        self.__size = dim[0] * dim[1]
        # length of the output signal y
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

        self.__variance = variance
        self.__noise = None
        self.gaussian_noise()

        if measurement_operator is not None:
            assert measurement_operator.shape[1] == self.__size, "Measurement operator shape does not match dim"
            self.__measurement_operator = measurement_operator
        else:
            self.random_measurement_operator(self.__y_size)

        fig, ((self.__ax1, self.__ax2, self.__ax3), (self.__ax4, self.__ax5, self.__ax6)) = plt.subplots(2, 3)
        fig.suptitle("Spare + Smooth Signal")

    @property
    def dim(self) -> Tuple[int, int]:
        return self.__dim

    @property
    def sparse(self) -> np.ndarray:
        return self.__sparse

    @property
    def smooth(self) -> np.ndarray:
        return self.__smooth

    @property
    def measurement_operator(self) -> np.ndarray:
        return self.__measurement_operator

    @property
    def x(self) -> np.ndarray:
        return self.__sparse + self.__smooth

    @property
    def H(self) -> np.ndarray:
        return self.measurement_operator

    @property
    def y0(self) -> np.ndarray:
        return self.H @ self.x.ravel()

    @property
    def y(self) -> np.ndarray:
        return self.y0 + self.__noise

    @property
    def noise(self) -> np.ndarray:
        return self.__noise

    def random_sparse(self) -> None:
        """
        Creates a new random sparse component with a density of 5%
        """
        self.__sparse = sp.random(self.__dim[0], self.__dim[1], density=0.05, data_rvs=np.random.randn).toarray()

    def random_smooth(self) -> None:
        """
        Creates a new random smooth component
        """
        self.__smooth = np.zeros(self.__dim)
        # number of gaussian we create
        nb = int(0.05 * self.__size)
        for i in range(nb):
            # Random center of a gaussian
            a, b = np.random.uniform(-1, 1), np.random.uniform(-1, 1)
            x, y = np.meshgrid(np.linspace(a - 1, a + 1, self.__dim[1]), np.linspace(b - 1, b + 1, self.__dim[0]))
            var = 1 - np.abs(np.random.normal(0, 1))
            g = np.exp(-((np.sqrt(x * x + y * y)) ** 2 / (2.0 * var ** 2)))
            self.__smooth += g
        self.__smooth / nb

    def random_measurement_operator(self, size: int) -> None:
        """
        Creates a new random measurement operator with size random lines of the DFT matrix

        Parameters
        ----------
        size :
            Numbers of lines of the DFT matrix we want to pick, witch is also the new dimension of y
        """
        assert self.__size >= size >= 0
        self.__y_size = size
        dft_mtx = dft(self.__size)
        rand = np.random.choice(self.__size, size, replace=False)
        self.__measurement_operator = dft_mtx[rand]
        self.gaussian_noise()

    def gaussian_noise(self, variance: np.float64 = None) -> None:
        """
        Creates a new gaussian white noise

        Parameters
        ----------
        variance :
            Variance of the gaussian white noise
            if None then we choose the last input
            if the variance was never changed we take 1
        """
        if variance is None:
            variance = self.__variance
        else:
            self.__variance = variance
        self.__noise = np.random.normal(0, variance, self.__y_size)

    def plot(self) -> None:
        """
        Plot all signals in 2d
        """
        self.__ax1.imshow(self.x)
        self.__ax1.set_title("X")
        self.__ax2.imshow(self.smooth)
        self.__ax2.set_title("Smooth")
        self.__ax3.imshow(self.sparse)
        self.__ax3.set_title("Sparse")
        self.__ax4.plot(self.y.reshape(self.__dim))
        self.__ax4.set_title("Y")
        self.__ax5.plot(self.y0.reshape(self.__dim))
        self.__ax5.set_title("Y0")
        self.__ax6.imshow(self.noise.reshape(self.__dim))
        self.__ax6.set_title("Noise")
        plt.show()
