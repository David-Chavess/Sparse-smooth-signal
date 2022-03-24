from __future__ import annotations

from typing import List

import numpy as np
from matplotlib import pyplot as plt
from pycsou.core import LinearOperator

from src.lasso_solver import LassoSolver
from src.solver import MyMatrixFreeOperator
from src.sparse_smooth_signal import SparseSmoothSignal
from src.sparse_smooth_solver import SparseSmoothSolver
from src.tikhonov_solver import TikhonovSolver


def loss(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    return np.sum((x1 - x2) ** 2)


def print_best(loss_x, loss_x1, loss_x2) -> None:
    print("Sparse + Smooth:")
    for x in sorted(loss_x, key=loss_x.get):
        print(f"{x} : {loss_x[x]}")
    print("Sparse:")
    for x in sorted(loss_x1, key=loss_x1.get):
        print(f"{x} : {loss_x1[x]}")
    print("Smooth:")
    for x in sorted(loss_x2, key=loss_x2.get):
        print(f"{x} : {loss_x2[x]}")


def plot_3(x: np.ndarray, x_tik: np.ndarray, x_lasso: np.ndarray, name: str = "") -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.canvas.manager.set_window_title(f'Spare + Smooth Signal : {name}')
    fig.suptitle(name)

    im = ax1.imshow(x, vmin=0, vmax=6)
    ax1.set_axis_off()
    ax1.set_title("Spare + Smooth")

    im = ax2.imshow(x_tik, vmin=0, vmax=6)
    ax2.set_axis_off()
    ax2.set_title("Smooth")

    im = ax3.imshow(x_lasso, vmin=0, vmax=6)
    ax3.set_axis_off()
    ax3.set_title("Lasso")

    fig.subplots_adjust(bottom=0.05, top=0.9, left=0.01, right=0.95,
                        wspace=0.02)
    cb_ax = fig.add_axes([0.95, 0.05, 0.02, 0.85])
    cbar = fig.colorbar(im, cax=cb_ax)


def plot_4(x_sparse: np.ndarray, x_smooth: np.ndarray, x1: np.ndarray, x2: np.ndarray, name: str = "") -> None:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
    fig.canvas.manager.set_window_title(f'Spare + Smooth Signal : {name}')
    fig.suptitle(name)

    im_p = ax1.imshow(x_sparse, vmin=0, vmax=7)
    ax1.axis('off')
    ax1.set_title("Original Sparse")

    im_s = ax2.imshow(x_smooth, vmin=0, vmax=1)
    ax2.axis('off')
    ax2.set_title("Original Smooth")

    im = ax3.imshow(x1, vmin=0, vmax=7)
    ax3.axis('off')
    ax3.set_title("Reconstructed Sparse")

    im = ax4.imshow(x2, vmin=0, vmax=1)
    ax4.axis('off')
    ax4.set_title("Reconstructed Smooth")

    fig.subplots_adjust(bottom=0.05, top=0.9, left=0.01, right=0.95,
                        wspace=0.1, hspace=0.1)
    cb_ax = fig.add_axes([0.95, 0.05, 0.02, 0.85])
    cbar = fig.colorbar(im_s, cax=cb_ax)
    cb_ax = fig.add_axes([0.45, 0.05, 0.02, 0.85])
    cbar = fig.colorbar(im_p, cax=cb_ax)


def test(s: SparseSmoothSignal, l1: float, l2: float, op: None | str | LinearOperator, name: str):
    solver = SparseSmoothSolver(s.y, s.H, l1, l2, op)
    x1, x2 = solver.solve()

    x1 = x1.reshape(s.dim)
    x2 = x2.reshape(s.dim)
    x = x1 + x2

    plot_4(s.sparse, s.smooth, x1, x2, name)

    return loss(s.x, x), loss(s.sparse, x1), loss(s.smooth, x2)


def test_solvers(s: SparseSmoothSignal, lambda1: (float, float), lambda2: (float, float | LinearOperator)):
    sol = SparseSmoothSolver(s.y, s.H, lambda1[0], lambda2[0], "deriv1")
    x_ss = sol.solve()
    x_ss = (x_ss[0]+x_ss[1]).reshape(dim)

    sol = TikhonovSolver(s.y, s.H, lambda2[1])
    _, x_tik = sol.solve()
    x_tik = x_tik.reshape(dim)

    sol = LassoSolver(s.y, s.H, lambda1[1])
    x_lasso, _ = sol.solve()
    x_lasso = x_lasso.reshape(dim)

    plot_3(x_ss, x_tik, x_lasso)

    print(f"Sparse + Smooth loss : {loss(s.x, x_ss)}")
    print(f"Tik loss : {loss(s.x, x_tik)}")
    print(f"Lasso loss : {loss(s.x, x_lasso)}")
    s.show()


def test_hyperparameters(s: SparseSmoothSignal, L: List[float], lambda1: List[float], lambda2: List[float],
                         operators: List[None | str | LinearOperator], psnr: List[float]):
    size = s.dim[0] * s.dim[1]
    loss_x = {}
    loss_x1 = {}
    loss_x2 = {}
    for m in L:
        op = MyMatrixFreeOperator(s.dim, int(m * size))
        s.H = op
        for p in psnr:
            s.gaussian_noise(p)
            for l1 in lambda1:
                for l2 in lambda2:
                    for op in operators:
                        name = f"Lambda1:{l1}, Lambda2:{l2}, H:{m * 100}% measurements, psnr:{p}, l2 operator:{op}"
                        loss_x[name], loss_x1[name], loss_x2[name] = test(s, l1, l2, op, name)

    print_best(loss_x, loss_x1, loss_x2)
    s.show()


def test_lambda(s: SparseSmoothSignal, L: float, lambdas: List[float], thetas: List[float],
                operator: None | str | LinearOperator = None, psnr: float = 50.):

    op = MyMatrixFreeOperator(s.dim, int(L * s.dim[0] * s.dim[1]))
    s.H = op
    s.gaussian_noise(psnr)

    loss_x = {}
    loss_x1 = {}
    loss_x2 = {}
    for l in lambdas:
        for t in thetas:
            name = f"Lambda:{l}, Theta:{t}, H:{L * 100}% measurements, psnr:{psnr}, l2 operator:{operator}"
            loss_x[name], loss_x1[name], loss_x2[name] = test(s, l * t, l * (1 - t), operator, name)

    print_best(loss_x, loss_x1, loss_x2)
    s.show()


if __name__ == '__main__':
    dim = (64, 64)
    opf = MyMatrixFreeOperator(dim)
    s1 = SparseSmoothSignal(dim, measurement_operator=opf)
    test_lambda(s1, 0.5, [0.05, 0.1, 0.2, 0.3], [0.1, 0.25, 0.5], "D", 50.)
    # test_solvers(s1, (0.1, 0.1), (0.1, 0.1))
    # test_hyperparameters(s1, [0.2], ([0.1], [0.1, 0.25, 0.5, 0.75]), ["D"], [50.0])
