from __future__ import annotations

from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from pycsou.core import LinearOperator
from pycsou.linop import Gradient

from src.lasso_solver import LassoSolver
from src.solver import MyMatrixFreeOperator
from src.sparse_smooth_signal import SparseSmoothSignal
from src.sparse_smooth_solver import SparseSmoothSolver
from src.tikhonov_solver import TikhonovSolver


def nmse(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((x1 - x2) ** 2) / np.mean(x1 ** 2))


def get_D(D: str, shape: Tuple[int, int]) -> LinearOperator:
    grad = Gradient(shape, kind='forward')
    grad.compute_lipschitz_cst(tol=1e-3)
    if D == "D":
        op = grad
    # elif D == "D2":
    # TODO
    return op


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


def random_points(dim: Tuple[int, int], size: int) -> np.ndarray:
    points = set()

    while len(points) < size:
        # 2d Gaussian
        p = np.abs(np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], size - len(points)))
        p /= np.max(p)

        # Cast to index
        p[:, 0] *= dim[0] - 1
        p[:, 1] *= dim[1] - 1

        # Add to set as int types
        s = set(map(tuple, p.astype((int, int))))
        points.update(s)
        # We don't want to point (0, 0)
        points.discard((0, 0))

    # Plot distribution of points
    # v = np.array(sorted(list(points)))
    # plt.scatter(*zip(*v))
    # plt.show()

    return np.array(list(points))


def random_lines(dim: Tuple[int, int], size: int) -> np.ndarray:
    points = random_points(dim, size)
    index = points[:, 0] + points[:, 1] * dim[0]
    return index


def plot_3(x: np.ndarray, x_tik: np.ndarray, x_lasso: np.ndarray, name: str = "") -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.canvas.manager.set_window_title(f'Spare + Smooth Signal : {name}')
    fig.suptitle(name)

    im = ax1.imshow(x, vmin=0, vmax=7)
    ax1.set_axis_off()
    ax1.set_title("Spare + Smooth")

    im = ax2.imshow(x_tik, vmin=0, vmax=7)
    ax2.set_axis_off()
    ax2.set_title("Tikhonov")

    im = ax3.imshow(x_lasso, vmin=0, vmax=7)
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


def test(s: SparseSmoothSignal, l1: float, l2: float, op: None | LinearOperator, name: str):
    m = np.max(np.abs(s.H.adjoint(s.y)))
    solver = SparseSmoothSolver(s.y, s.H, l1 * m, l2 * m, op)
    x1, x2 = solver.solve()

    x1 = x1.reshape(s.dim)
    x2 = x2.reshape(s.dim)
    x = x1 + x2

    plot_4(s.sparse, s.smooth, x1, x2, name)

    return nmse(s.x, x), nmse(s.sparse, x1), nmse(s.smooth, x2)


def test_solvers(s: SparseSmoothSignal, lambda1: float, lambda2: float, operator: None | str | LinearOperator = None):
    if isinstance(operator, str):
        D = get_D(operator, s.dim)
    else:
        D = operator

    m = np.max(np.abs(s.H.adjoint(s.y)))
    l1 = lambda1 * m
    l2 = lambda2 * m

    sol = SparseSmoothSolver(s.y, s.H, l1, l2, D)
    x_ss = sol.solve()
    x_ss = (x_ss[0] + x_ss[1]).reshape(s.dim)

    sol = TikhonovSolver(s.y, s.H, l2)
    _, x_tik = sol.solve()
    x_tik = x_tik.reshape(s.dim)

    sol = LassoSolver(s.y, s.H, l1)
    x_lasso, _ = sol.solve()
    x_lasso = x_lasso.reshape(s.dim)

    plot_3(x_ss, x_tik, x_lasso)

    print(f"Sparse + Smooth loss : {nmse(s.x, x_ss)}")
    print(f"Tikhonov loss : {nmse(s.x, x_tik)}")
    print(f"Lasso loss : {nmse(s.x, x_lasso)}")
    s.show()


def test_hyperparameters(s: SparseSmoothSignal, H: List[float], lambdas: List[float], thetas: List[float],
                         operators: List[None | str | LinearOperator], psnr: List[float]):
    size = s.dim[0] * s.dim[1]
    loss_x = {}
    loss_x1 = {}
    loss_x2 = {}
    for op in operators:
        if isinstance(op, str):
            D = get_D(op, s.dim)
        else:
            D = op
        for h in H:
            s.H = MyMatrixFreeOperator(s.dim, int(h * size))
            for p in psnr:
                s.gaussian_noise(p)
                for l in lambdas:
                    for t in thetas:
                        name = f"λ:{l:.2f}, θ:{t:.2f}, {h:.1%} measurements, PSNR:{p:.0f}, l2 operator:{op.__str__()} "
                        loss_x[name], loss_x1[name], loss_x2[name] = test(s, l * t, l * (1 - t), D, name)

    print_best(loss_x, loss_x1, loss_x2)
    s.show()


def test_lambda(s: SparseSmoothSignal, L: float, lambdas: List[float], thetas: List[float],
                operator: None | str | LinearOperator = None, psnr: float = 50.):

    s.H = MyMatrixFreeOperator(s.dim, random_lines(s.dim, int(L * s.dim[0] * s.dim[1])))
    s.gaussian_noise(psnr)
    if isinstance(operator, str):
        D = get_D(operator, s.dim)
    else:
        D = operator

    loss_x = {}
    loss_x1 = {}
    loss_x2 = {}
    for l in lambdas:
        for t in thetas:
            name = f"λ:{l:.2f}, θ:{t:.2f}, {L:.1%} measurements, PSNR:{psnr:.0f}, l2 operator:{operator.__str__()}"
            loss_x[name], loss_x1[name], loss_x2[name] = test(s, l * t, l * (1 - t), D, name)

    print_best(loss_x, loss_x1, loss_x2)
    s.show()


if __name__ == '__main__':
    d = (128, 128)
    seed = 11
    s1 = SparseSmoothSignal(d)
    s1.random_sparse(seed)
    s1.random_smooth(seed)

    # L = 0.5
    #
    # lines = random_lines(s1.dim, int(L * s1.dim[0] * s1.dim[1]))
    #
    # fig, axes = plt.subplots(1, 2)
    # axes[0].hist(lines, bins=30)
    # o = MyMatrixFreeOperator(s1.dim, int(L * s1.dim[0] * s1.dim[1]))
    # s1.H = o
    # axes[1].hist(o.rand_lines, bins=30)
    # plt.show()
    #

    test_lambda(s1, 0.25, [0.1], [0.1], "D")

    # s1.H = MyMatrixFreeOperator(dim, int(0.2 * dim[0] * dim[1]))
    # test_solvers(s1, 0.1*0.1, 0.1*0.9, "D")

    # test_hyperparameters(s1, [0.35], [0.1], [0.05], [None, "D", "D2"], [50.0])
