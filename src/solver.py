from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Number
from typing import Union, Tuple

import numpy as np
from dask import array as da
from pycsou.core import LinearOperator
from pycsou.linop import DenseLinearOperator


class Solver(ABC):
    """
    Basic abstract class for all solvers.

    Any instance/subclass of this class must implement the abstract method solve()
    """

    def __init__(self, y: np.ndarray, operator: LinearOperator) -> None:
        """
        Parameters
        ----------
        y: np.ndarray
            Measurements y used in the inverse problem obtained by the linear measurement operator.
        operator: LinearOperator
            Linear operator used for measurements.
        """
        super().__init__()
        self.y = y
        self.operator = operator

    @abstractmethod
    def solve(self) -> Tuple[None | np.ndarray, None | np.ndarray]:
        """
        Run the solver

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Solver outcome. Two flatten images (x1, x2) where x1 is the sparse part and x2 the smooth one.
        """
        pass


class MyOperator(DenseLinearOperator):
    """My linear operator used in the solver, build by a complex matrix where in the adjoint method we ignore
    the imaginary part, because our solver works only with real numbers"""

    def adjoint(self, y: Union[Number, np.ndarray, da.core.Array]) -> Union[Number, np.ndarray]:
        return super().adjoint(y).real


class MyMatrixFreeOperator(LinearOperator):
    """A linear operator that is matrix free, so that it can much bigger. This operator is a Two dimensional Fourier
    Transform. It uses the fft2 from numpy in forward mode and ifft2 in adjoint mode."""

    def __init__(self, dim: Tuple[int, int], lines: None | int | np.ndarray = None):
        """
        Parameters
        ----------
        dim: Tuple[int, int]
            dimension if the image
        lines: None | int | np.ndarray
            Samples kept.
            If None we keep all samples.
            If int we keep the specified number of samples chosen randomly.
            If np.ndarray, it is list of the sample we want to use, we specified the index each sample
            from 0 to dim[0] * dim[1].
        """
        size = dim[0] * dim[1]
        shape = (size, size)
        self.dim = dim
        self.rand_lines = None
        if isinstance(lines, int) and lines != shape[0] * shape[1]:
            assert 0 < lines <= shape[0] * shape[1], "invalid numbers of lines"
            shape = (lines, size)
            self.rand_lines = np.sort(np.random.choice(size, lines, replace=False))
        elif isinstance(lines, np.ndarray):
            shape = (lines.size, size)
            self.rand_lines = lines
        super().__init__(shape, dtype=None, is_explicit=False, is_dense=False, is_sparse=False, is_dask=False,
                         is_symmetric=False, lipschitz_cst=1.)

    def __call__(self, arg: Number | np.ndarray) -> Number | np.ndarray:
        y = np.fft.fft2(arg.reshape(self.dim), norm='ortho').ravel()

        if self.rand_lines is None:
            return y
        else:
            return y[self.rand_lines]

    def adjoint(self, y: Number | np.ndarray) -> Number | np.ndarray:

        if self.rand_lines is not None:
            y_big = np.zeros(self.dim, dtype=np.complex128).ravel()
            y_big[self.rand_lines] = y
        else:
            y_big = y

        x = np.fft.ifft2(y_big.reshape(self.dim), norm='ortho')
        return x.ravel().real

