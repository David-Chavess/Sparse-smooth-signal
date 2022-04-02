from __future__ import annotations

from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from pycsou.core import LinearOperator
from pycsou.linop import Gradient, Laplacian
from scipy.stats import wasserstein_distance

from src.lasso_solver import LassoSolver
from src.solver import MyMatrixFreeOperator
from src.sparse_smooth_signal import SparseSmoothSignal
from src.sparse_smooth_solver import SparseSmoothSolver
from src.tikhonov_solver import TikhonovSolver


def nmse(x1: np.ndarray, x2: np.ndarray) -> float:
    return np.sqrt(np.mean((x1 - x2) ** 2) / np.mean(x1 ** 2))


def Wasserstein_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    return wasserstein_distance(x1.ravel(), x2.ravel())


def get_L2_operator(dim: Tuple[int, int], op_l2: None | str | LinearOperator) -> LinearOperator:
    if isinstance(op_l2, str):
        if op_l2 == "D":
            op = Gradient(dim, kind='forward')
        elif op_l2 == "L":
            op = Laplacian(dim)
        op.compute_lipschitz_cst(tol=1e-3)
        return op
    else:
        return op_l2


def get_MyMatrixFreeOperator(dim: Tuple[int, int], L: float) -> MyMatrixFreeOperator:
    return MyMatrixFreeOperator(dim, random_lines(dim, int(L * dim[0] * dim[1])))


def off_set_smooth(smooth: np.ndarray, x2: np.ndarray) -> np.ndarray:
    return np.max(smooth) - np.max(x2)  # np.abs(np.min(x2))


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
        p = np.abs(np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], size - len(points) + 1))
        p /= np.max(p)

        # Cast to index
        p[:, 0] *= dim[0] - 1
        p[:, 1] *= dim[1] - 1

        # Add to set as int types
        s = set(map(tuple, p.astype((int, int))))
        points.update(s)

    # We don't want the point (0, 0)
    points.discard((0, 0))
    # points.add((0, 0))

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
    ax1.set_title("Sparse + Smooth")

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

    fig.subplots_adjust(bottom=0.05, top=0.9, left=0.01, right=0.92,
                        wspace=0.1, hspace=0.1)
    cb_ax = fig.add_axes([0.92, 0.05, 0.02, 0.85])
    cbar = fig.colorbar(im_s, cax=cb_ax)
    cb_ax = fig.add_axes([0.45, 0.05, 0.02, 0.85])
    cbar = fig.colorbar(im_p, cax=cb_ax)


def plot_loss(x, loss_x1, loss_x2, name: str = "", var: str = ""):
    fig, (ax1, ax2), = plt.subplots(1, 2, figsize=(10, 5))
    fig.canvas.manager.set_window_title(f'Loss comparison')
    fig.suptitle(name)

    ax1.plot(x, loss_x1)
    ax1.set_xlabel(var)
    ax1.set_ylabel("Wasserstein_distance")
    ax1.set_title("L1 loss")

    ax2.plot(x, loss_x2)
    ax2.set_xlabel(var)
    ax2.set_ylabel("NMSE")
    ax2.set_title("L2 loss")


def test(s: SparseSmoothSignal, l1: float, l2: float, op: None | LinearOperator) -> Tuple[np.ndarray, np.ndarray]:
    m = np.max(np.abs(s.H.adjoint(s.y)))
    solver = SparseSmoothSolver(s.y, s.H, l1 * m, l2 * m, op)
    x1, x2 = solver.solve()

    x1 = x1.reshape(s.dim)
    x2 = x2.reshape(s.dim)

    # print(f"x1 min: {np.min(x1)}")
    # print(f"x2 min: {np.min(x2)}")
    # print(f"x1 nb min: {(x1 < 0).sum()}")
    # print(f"x2 nb min: {(x2 < 0).sum()}")

    x2 += off_set_smooth(s.smooth, x2)

    x1[x1 < 0] = 0

    # print(f"x2 min: {np.min(x2)}")
    # print(f"x2 max: {np.max(x2)}")
    return x1, x2


def test_solvers(s: SparseSmoothSignal, lambda1: float, lambda2: float,
                 operator_l2: None | str | LinearOperator = 'L'):
    op = get_L2_operator(s.dim, operator_l2)

    m = np.max(np.abs(s.H.adjoint(s.y)))
    l1 = lambda1 * m
    l2 = lambda2 * m

    sol = SparseSmoothSolver(s.y, s.H, l1, l2, op)
    x_ss = sol.solve()
    x_ss = (x_ss[0] + x_ss[1] + off_set_smooth(s.smooth, x_ss[1])).reshape(s.dim)

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


def test_hyperparameters(s: SparseSmoothSignal, L: List[float], lambdas: List[float], thetas: List[float],
                         operators_l2: List[None | str | LinearOperator], psnr: List[float]):
    loss_x = {}
    loss_x1 = {}
    loss_x2 = {}
    for op in operators_l2:
        op_l2 = get_L2_operator(s.dim, op)
        for h in L:
            s.H = get_MyMatrixFreeOperator(s.dim, h)
            for p in psnr:
                s.gaussian_noise(p)
                for l in lambdas:
                    for t in thetas:
                        name = f"λ:{l:.2f}, θ:{t:.2f}, {h:.1%} measurements, PSNR:{p:.0f}, l2 operator:{op.__str__()}"
                        x1, x2 = test(s, l * t, l * (1 - t), op_l2)
                        plot_4(s.sparse, s.smooth, x1, x2, name)
                        loss_x[name] = nmse(s.x, x1 + x2)
                        loss_x1[name] = Wasserstein_distance(s.sparse, x1)
                        loss_x2[name] = nmse(s.smooth, x2)

    print_best(loss_x, loss_x1, loss_x2)
    s.show()


def test_numbers_of_measurements(s: SparseSmoothSignal, L_min: float, L_max: float, nb: int, lambda_: float,
                                 theta: float, operator_l2: None | str | LinearOperator = 'L', psnr: float = 50.):
    op_l2 = get_L2_operator(s.dim, operator_l2)
    s.psnr = psnr

    loss_x1 = []
    loss_x2 = []
    measurements = np.linspace(L_min, L_max, nb)
    for l in measurements:
        s.H = get_MyMatrixFreeOperator(s.dim, l)
        x1, x2 = test(s, lambda_ * theta, lambda_ * (1 - theta), op_l2)
        loss_x1.append(Wasserstein_distance(s.sparse, x1))
        loss_x2.append(nmse(s.smooth, x2))

    name = f"λ:{lambda_:.2f}, θ:{theta:.2f}, PSNR:{psnr:.0f}, l2 operator:{operator_l2.__str__()}"
    print(f"Best value L1: {measurements[np.argmin(loss_x1)]}")
    print(f"Best value L2: {measurements[np.argmin(loss_x2)]}")
    plot_loss(measurements * 100, loss_x1, loss_x2, name, "Numbers of measurements")


def test_thetas(s: SparseSmoothSignal, theta_min: float, theta_max: float, nb: int, L: float, lambda_: float,
                operator_l2: None | str | LinearOperator = 'L', psnr: float = 50.):
    s.H = get_MyMatrixFreeOperator(s.dim, L)
    s.gaussian_noise(psnr)
    op_l2 = get_L2_operator(s.dim, operator_l2)

    loss_x1 = []
    loss_x2 = []
    thetas = np.linspace(theta_min, theta_max, nb)
    for t in thetas:
        x1, x2 = test(s, lambda_ * t, lambda_ * (1 - t), op_l2)
        loss_x1.append(Wasserstein_distance(s.sparse, x1))
        loss_x2.append(nmse(s.smooth, x2))

    name = f"λ:{lambda_:.2f}, {L:.1%} measurements, PSNR:{psnr:.0f}, l2 operator:{operator_l2.__str__()}"
    print(f"Best value L1: {thetas[np.argmin(loss_x1)]}")
    print(f"Best value L2: {thetas[np.argmin(loss_x2)]}")
    plot_loss(thetas, loss_x1, loss_x2, name, "θ")

    t = thetas[np.argmin(loss_x2)]
    name = f"λ:{lambda_:.2f}, θ:{t:.2f}, {L:.1%} measurements, PSNR:{psnr:.0f}, l2 operator:{operator_l2.__str__()}"
    x1, x2 = test(s, lambda_ * t, lambda_ * (1 - t), op_l2)
    plot_4(s.sparse, s.smooth, x1, x2, name)
    s.show()


def test_lambdas(s: SparseSmoothSignal, lambda_min: float, lambda_max: float, nb: int, L: float, theta: float,
                 operator_l2: None | str | LinearOperator = 'L', psnr: float = 50.):
    s.H = get_MyMatrixFreeOperator(s.dim, L)
    s.gaussian_noise(psnr)
    op_l2 = get_L2_operator(s.dim, operator_l2)

    loss_x1 = []
    loss_x2 = []
    lambdas = np.linspace(lambda_min, lambda_max, nb)
    for l in lambdas:
        x1, x2 = test(s, l * theta, l * (1 - theta), op_l2)
        loss_x1.append(Wasserstein_distance(s.sparse, x1))
        loss_x2.append(nmse(s.smooth, x2))

    name = f"θ:{theta:.2f}, {L:.1%} measurements, PSNR:{psnr:.0f}, l2 operator:{operator_l2.__str__()}"
    print(f"Best value L1: {lambdas[np.argmin(loss_x1)]}")
    print(f"Best value L2: {lambdas[np.argmin(loss_x2)]}")
    plot_loss(lambdas, loss_x1, loss_x2, name, "λ")

    l = lambdas[np.argmin(loss_x1)]
    name = f"λ:{l:.2f}, θ:{theta:.2f}, {L:.1%} measurements, PSNR:{psnr:.0f}, l2 operator:{operator_l2.__str__()}"
    x1, x2 = test(s, l * theta, l * (1 - theta), op_l2)
    plot_4(s.sparse, s.smooth, x1, x2, name)
    s.show()


def test_noise(s: SparseSmoothSignal, psnr_min: float, psnr_max: float, nb: int, L: float, lambda_: float, theta: float,
               operator_l2: None | str | LinearOperator = 'L'):
    s.H = get_MyMatrixFreeOperator(s.dim, L)
    op_l2 = get_L2_operator(s.dim, operator_l2)

    loss_x1 = []
    loss_x2 = []
    psnrs = np.linspace(psnr_min, psnr_max, nb)
    for p in psnrs:
        s.gaussian_noise(p)
        x1, x2 = test(s, lambda_ * theta, lambda_ * (1 - theta), op_l2)
        loss_x1.append(Wasserstein_distance(s.sparse, x1))
        loss_x2.append(nmse(s.smooth, x2))

    name = f"λ:{lambda_:.2f}, θ:{theta:.2f}, {L:.1%} measurements, l2 operator:{operator_l2.__str__()}"
    print(f"Best value L1: {psnrs [np.argmin(loss_x1)]}")
    print(f"Best value L2: {psnrs [np.argmin(loss_x2)]}")
    plot_loss(psnrs, loss_x1, loss_x2, name, "PSNR")


if __name__ == '__main__':
    d = (64, 64)
    seed = 11
    s1 = SparseSmoothSignal(d)
    sp = s1.sparse
    s1.random_sparse(seed)
    s1.random_smooth(seed)

    # print(Wasserstein_distance(sp.ravel(), sp.ravel()))

    # test_numbers_of_measurements(s1, 0.1, 0.75, 25, 0.1, 0.1, "L", 40.)
    # test_thetas(s1, 0.005, 0.8, 10, 0.4, 0.2, "L", 40.)
    # test_lambdas(s1, 0.01, 0.5, 25, 0.4, 0.1, "L", 40.)
    # test_noise(s1, 0., 50., 25, 0.25, 0.1, 0.1, "L")

    # L = 0.05
    # random_points(d, int(L * s1.dim[0] * s1.dim[1]))
    # lines = random_lines(s1.dim, int(L * s1.dim[0] * s1.dim[1]))
    #
    # fig, axes = plt.subplots(1, 2)
    # axes[0].hist(lines, bins=30)
    # o = MyMatrixFreeOperator(s1.dim, int(L * s1.dim[0] * s1.dim[1]))
    # s1.H = o
    # axes[1].hist(o.rand_lines, bins=30)
    # plt.show()

    # s1.H = MyMatrixFreeOperator(d, int(0.2 * d[0] * d[1]))
    # test_solvers(s1, 0.1 * 0.1, 0.1 * 0.9, "L")

    # test_hyperparameters(s1, [0.2, 0.3, 0.4], [0.1], [0.1], ["L"], [30.0])
