from abc import ABC, abstractmethod
from numbers import Number
from typing import Union

import numpy as np
from dask import array as da
from pycsou.linop import DenseLinearOperator


class Solver(ABC):

    def __init__(self, y: np.ndarray, operator: np.ndarray) -> None:
        super().__init__()
        self.y = y
        self.operator = operator

    @abstractmethod
    def solve(self) -> (np.ndarray, np.ndarray):
        pass


class MyOperator(DenseLinearOperator):

    def adjoint(self, y: Union[Number, np.ndarray, da.core.Array]) -> Union[Number, np.ndarray]:
        return np.real(super().adjoint(y))
