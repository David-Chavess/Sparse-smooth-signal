from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.linalg import dft
from scipy import sparse
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
    plot_sparse() -> None
        Plot the sparse component
    plot_smooth() -> None
        Plot the smooth component
    plot_x() -> None
        Plot the signal x
    plot_y0() -> None
        Plot the signal y0
    plot_y() -> None
        Plot the signal y
    plot_noise() -> None
        Plot the noise
    show() -> None
        Show the plotted signals, it is used after all draw_?() functions
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
            self.__sparse = sparse.ravel()
        else:
            self.random_sparse()

        if smooth is not None:
            assert smooth.shape == dim, "Smooth is not the same size as dim"
            self.__smooth = smooth.ravel()
        else:
            self.random_smooth()

        if measurement_operator is not None:
            assert measurement_operator.shape[1] == self.__size, "Measurement operator shape does not match dim"
            self.__measurement_operator = measurement_operator
        else:
            self.random_measurement_operator(self.__y_size)

        self.__variance = variance
        self.__noise = None
        self.gaussian_noise()

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
        if self.__measurement_operator is not None:
            return self.H @ self.x
        return None

    @property
    def y(self) -> np.ndarray:
        if self.__noise is not None:
            return self.y0 + self.__noise
        return None

    @property
    def noise(self) -> np.ndarray:
        return self.__noise

    def random_sparse(self) -> None:
        """
        Creates a new random sparse component with a density of 25%
        """
        self.__sparse = sparse.random(self.__dim[0], self.__dim[1], density=0.25, data_rvs=np.random.randn)\
            .toarray().ravel()

    def random_smooth(self) -> None:
        """
        Creates a new random smooth component
        """
        amp = np.random.randint(0, 5, 2)
        freq = 2*np.pi*np.random.rand(2)
        t = np.linspace(0, self.__size/10, self.__size)
        self.__smooth = amp[0]*np.sin(t/freq[0]) + amp[1]*np.cos(t/freq[1])

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
        self.__noise = None

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

    def plot_sparse(self) -> None:
        """
        Plot the sparse component
        """
        plt.plot(self.sparse, label="sparse")

    def plot_smooth(self) -> None:
        """
        Plot the smooth component
        """
        plt.plot(self.smooth, label="smooth")

    def plot_x(self) -> None:
        """
        Plot the signal x
        """
        plt.plot(self.x, label="x")

    def plot_y0(self) -> None:
        """
        Plot the signal y0
        """
        plt.plot(self.y0, label="y0")

    def plot_y(self) -> None:
        """
        Plot the signal y
        """
        plt.plot(self.y, label="y")

    def plot_noise(self) -> None:
        """
        Plot the noise
        """
        plt.plot(self.noise, label="noise")

    @staticmethod
    def show() -> None:
        """
        Show the plotted signals, it is used after all draw_?() functions
        """
        plt.title("Spare + Smooth Signal")
        plt.ylabel("Amplitude")
        plt.legend(loc="best")
        plt.show()
