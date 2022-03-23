from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Number
from typing import Union, Tuple

import numpy as np
from dask import array as da
from pycsou.core import LinearOperator
from pycsou.linop import DenseLinearOperator


class Solver(ABC):

    def __init__(self, y: np.ndarray, operator: LinearOperator) -> None:
        super().__init__()
        self.y = y
        self.operator = operator

    @abstractmethod
    def solve(self) -> (np.ndarray, np.ndarray):
        pass


class MyOperator(DenseLinearOperator):

    def adjoint(self, y: Union[Number, np.ndarray, da.core.Array]) -> Union[Number, np.ndarray]:
        return super().adjoint(y).real


class MyMatrixFreeOperator(LinearOperator):

    def __init__(self, dim: Tuple[int, int], lines: None | int | np.ndarray = None):
        shape = (dim[0] * dim[1], dim[0] * dim[1])
        self.dim = dim
        self.rand_lines = None
        if isinstance(lines, int) and lines != shape[0] * shape[1]:
            assert 0 < lines <= shape[0] * shape[1], "invalid numbers of lines"
            shape = (lines, dim[0] * dim[1])
            self.rand_lines = np.sort(np.random.choice(shape[0] * shape[1], lines, replace=False))
        elif isinstance(lines, np.ndarray):
            shape = (lines.size, dim[0] * dim[1])
            self.rand_lines = lines
        super().__init__(shape, dtype=None, is_explicit=False, is_dense=False, is_sparse=False, is_dask=False,
                         is_symmetric=False, lipschitz_cst=1)

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

    def compute_lipschitz_cst(self, **kwargs):
        return self.lipschitz_cst
