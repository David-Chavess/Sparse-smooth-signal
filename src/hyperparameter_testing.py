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
    return wasserstein_distance(x1.ravel() / np.sum(x1), x2.ravel() / np.sum(x2))


def get_L2_operator(dim: Tuple[int, int], op_l2: None | str | LinearOperator) -> LinearOperator:
    if isinstance(op_l2, str):
        if op_l2 == "Gradient":
            op = Gradient(dim, kind='forward')
        elif op_l2 == "Laplacian":
            op = Laplacian(dim)
        else:
            raise ValueError("Operator name is invalid")
        op.compute_lipschitz_cst(tol=1e-3)
        return op
    else:
        return op_l2


def get_MyMatrixFreeOperator(dim: Tuple[int, int], L: float) -> MyMatrixFreeOperator:
    return MyMatrixFreeOperator(dim, random_lines(dim, int(L * dim[0] * dim[1])))


def off_set_smooth(smooth: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.mean(smooth) - np.mean(x)


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


def plot_2(x1: np.ndarray, x2: np.ndarray, x1_name: str = "", x2_name: str = "", name: str = "", min_range: float = 0,
           max_range: float = 7) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.canvas.manager.set_window_title(f'Spare + Smooth Signal : {name}')
    fig.suptitle(name)

    im = ax1.imshow(x1, vmin=min_range, vmax=max_range)
    ax1.set_axis_off()
    ax1.set_title(x1_name)

    im = ax2.imshow(x2, vmin=min_range, vmax=max_range)
    ax2.set_axis_off()
    ax2.set_title(x2_name)

    fig.subplots_adjust(bottom=0.05, top=0.9, left=0.01, right=0.95,
                        wspace=0.02)
    cb_ax = fig.add_axes([0.95, 0.05, 0.02, 0.85])
    cbar = fig.colorbar(im, cax=cb_ax)


def plot_solvers(x: np.ndarray, x_tik: np.ndarray, x_tik_op: np.ndarray, x_lasso: np.ndarray, name: str = "",
                 min_range: float = 0,
                 max_range: float = 7) -> None:
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
    fig.canvas.manager.set_window_title(f'Spare + Smooth Signal : {name}')
    fig.suptitle(name)

    im = ax1.imshow(x, vmin=min_range, vmax=max_range)
    ax1.set_axis_off()
    ax1.set_title("Sparse + Smooth")

    im = ax2.imshow(x_tik, vmin=min_range, vmax=max_range)
    ax2.set_axis_off()
    ax2.set_title("Tikhonov without operator")

    im = ax3.imshow(x_tik_op, vmin=min_range, vmax=max_range)
    ax3.set_axis_off()
    ax3.set_title("Tikhonov with operator")

    im = ax4.imshow(x_lasso, vmin=min_range, vmax=max_range)
    ax4.set_axis_off()
    ax4.set_title("Lasso")

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


def plot_8(x_sparse: np.ndarray, x_smooth: np.ndarray, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, x4: np.ndarray,
           x5: np.ndarray, x6: np.ndarray, name: str = "") -> None:
    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(20, 10))
    fig.canvas.manager.set_window_title(f'Spare + Smooth Signal : {name}')
    fig.suptitle(name)

    im_p = ax1.imshow(x_sparse, vmin=0, vmax=7)
    ax1.axis('off')
    ax1.set_title("Original")

    im = ax2.imshow(x1, vmin=0, vmax=7)
    ax2.axis('off')
    ax2.set_title("Best choose")

    im = ax3.imshow(x3, vmin=0, vmax=7)
    ax3.axis('off')
    ax3.set_title("Random")

    im = ax4.imshow(x5, vmin=0, vmax=7)
    ax4.axis('off')
    ax4.set_title("Random with more chance on low frequency")

    im_s = ax5.imshow(x_smooth, vmin=0, vmax=1)
    ax5.axis('off')

    im = ax6.imshow(x2, vmin=0, vmax=1)
    ax6.axis('off')

    im = ax7.imshow(x4, vmin=0, vmax=1)
    ax7.axis('off')

    im = ax8.imshow(x6, vmin=0, vmax=1)
    ax8.axis('off')

    fig.subplots_adjust(bottom=0.05, top=0.9, left=0.01, right=0.92,
                        wspace=0.1, hspace=0.1)
    cb_ax = fig.add_axes([0.93, 0.50, 0.02, 0.40])
    cbar = fig.colorbar(im_p, cax=cb_ax)
    cb_ax = fig.add_axes([0.93, 0.05, 0.02, 0.40])
    cbar = fig.colorbar(im_s, cax=cb_ax)


def plot_4_smooth(x_smooth: np.ndarray, x_None: np.ndarray, x_Gradian: np.ndarray, x_Laplacian: np.ndarray) -> None:
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))

    im = ax1.imshow(x_smooth, vmin=0, vmax=1)
    ax1.axis('off')
    ax1.set_title("Original Smooth")

    im = ax2.imshow(x_None)
    ax2.axis('off')
    ax2.set_title("Smooth without operator")

    im = ax3.imshow(x_Gradian, vmin=0, vmax=1)
    ax3.axis('off')
    ax3.set_title("Smooth with Gradient")

    im = ax4.imshow(x_Laplacian, vmin=0, vmax=1)
    ax4.axis('off')
    ax4.set_title("Smooth with Laplacian")

    fig.subplots_adjust(bottom=0.05, top=0.9, left=0.01, right=0.92,
                        wspace=0.1, hspace=0.1)
    cb_ax = fig.add_axes([0.93, 0.05, 0.02, 0.85])
    cbar = fig.colorbar(im, cax=cb_ax)


def plot_loss(x, loss_x1, loss_x2, name: str = "", var: str = ""):
    fig, (ax1, ax2), = plt.subplots(1, 2, figsize=(10, 5))
    fig.canvas.manager.set_window_title(f'Loss comparison')
    fig.suptitle(name)

    ax1.plot(x, loss_x1)
    ax1.set_xlabel(var)
    ax1.set_ylabel("Wasserstein distance")
    ax1.set_title("L1 loss")

    ax2.plot(x, loss_x2)
    ax2.set_xlabel(var)
    ax2.set_ylabel("NMSE")
    ax2.set_title("L2 loss")


def plot_picks(x, nb_picks, pick_found, wrong_picks_found, var: str = ""):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    fig.canvas.manager.set_window_title(f'Recovered pick comparison')
    ax.set_title("Recovered pick comparison")

    ax.semilogy(x, pick_found, label='Picks found')
    ax.semilogy(x, wrong_picks_found, label='Wrong Picks found')
    ax.axhline(y=nb_picks, label='number of picks', linestyle='--')
    ax.set_xlabel(var)
    ax.set_ylabel("Number of picks")
    ax.legend()


def test(s: SparseSmoothSignal, l1: float, l2: float, op: None | LinearOperator) -> Tuple[np.ndarray, np.ndarray]:
    m = np.max(np.abs(s.H.adjoint(s.y)))
    solver = SparseSmoothSolver(s.y, s.H, l1 * m, l2 * m, op)
    x1, x2 = solver.solve()

    x1 = x1.reshape(s.dim)
    x2 = x2.reshape(s.dim)

    x2 += off_set_smooth(s.smooth, x2)
    x1[x1 < 0] = 0

    return x1, x2


def test_solvers(s: SparseSmoothSignal, lambda1: float, lambda2: float,
                 operator_l2: None | str | LinearOperator = "Laplacian", name: str = ""):
    op = get_L2_operator(s.dim, operator_l2)

    x_sparse, x_smooth = test(s, lambda1, lambda2, op)
    x_sparse = x_sparse.reshape(s.dim)
    x_smooth = x_smooth.reshape(s.dim)

    plot_4(s.sparse, s.smooth, x_sparse, x_smooth, name)

    m = np.max(np.abs(s.H.adjoint(s.y)))
    lambda1 = lambda1 * m
    lambda2 = lambda2 * m

    sol = TikhonovSolver(s.y, s.H, lambda2)
    _, x_tik = sol.solve()
    x_tik = x_tik.reshape(s.dim)
    x_tik += off_set_smooth(s.smooth, x_tik)

    sol = TikhonovSolver(s.y, s.H, lambda2, op)
    _, x_tik_op = sol.solve()
    x_tik_op = x_tik_op.reshape(s.dim)
    x_tik_op += off_set_smooth(s.smooth, x_tik_op)

    sol = LassoSolver(s.y, s.H, lambda1)
    x_lasso, _ = sol.solve()
    x_lasso = x_lasso.reshape(s.dim)

    plot_solvers(x_sparse + x_smooth, x_tik, x_tik_op, x_lasso, name)

    picks_found(s.sparse, x_sparse, 1)
    picks_intensity(s.sparse, x_sparse)
    picks_found(s.sparse, x_tik, 1)
    picks_intensity(s.sparse, x_tik)
    picks_found(s.sparse, x_tik_op, 1)
    picks_intensity(s.sparse, x_tik_op)
    picks_found(s.sparse, x_lasso, 1)
    picks_intensity(s.sparse, x_lasso)
    return x_sparse, x_smooth, x_tik, x_tik_op, x_lasso


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
                        name = f"λ:{l:.2f}, θ:{t:.2f}, {h:.1%} measurements, PSNR:{p:.0f}, L2 operator:{op.__str__()}"
                        x1, x2 = test(s, l * t, l * (1 - t), op_l2)
                        plot_4(s.sparse, s.smooth, x1, x2, name)
                        loss_x[name] = nmse(s.x, x1 + x2)
                        loss_x1[name] = Wasserstein_distance(s.sparse, x1)
                        loss_x2[name] = nmse(s.smooth, x2)

    print_best(loss_x, loss_x1, loss_x2)
    s.show()


def test_numbers_of_measurements(s: SparseSmoothSignal, L_min: float, L_max: float, nb: int, lambda_: float,
                                 theta: float, operator_l2: None | str | LinearOperator = "Laplacian",
                                 psnr: float = 50.):
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

    name = f"λ:{lambda_:.2f}, θ:{theta:.2f}, PSNR:{psnr:.0f}, L2 operator:{operator_l2.__str__()}"
    print(f"Best value L1: {measurements[np.argmin(loss_x1)]}")
    print(f"Best value L2: {measurements[np.argmin(loss_x2)]}")
    plot_loss(measurements * 100, loss_x1, loss_x2, name, "Numbers of measurements")


def test_thetas(s: SparseSmoothSignal, theta_min: float, theta_max: float, nb: int, L: float, lambda_: float,
                operator_l2: None | str | LinearOperator = "Laplacian", psnr: float = 50.):
    s.H = get_MyMatrixFreeOperator(s.dim, L)
    s.gaussian_noise(psnr)
    op_l2 = get_L2_operator(s.dim, operator_l2)

    loss_x1 = []
    loss_x2 = []
    picks = []
    thetas = np.linspace(theta_min, theta_max, nb)
    for t in thetas:
        x1, x2 = test_best_lines(s, L, lambda_, t, psnr, op_l2)
        loss_x1.append(Wasserstein_distance(s.sparse, x1))
        loss_x2.append(nmse(s.smooth, x2))
        picks.append(picks_found(s.sparse, x1, 1))

    name = f"λ:{lambda_:.2f}, {L:.1%} measurements, PSNR:{psnr:.0f}, L2 operator:{operator_l2.__str__()}"
    print(f"Best value L1: {thetas[np.argmin(loss_x1)]}")
    print(f"Best value L2: {thetas[np.argmin(loss_x2)]}")
    plot_loss(thetas, loss_x1, loss_x2, name, "θ")
    picks = np.array(picks)
    plot_picks(thetas, len(np.argwhere(s.sparse >= 2)), picks[:, 0], picks[:, 1], "θ")


def test_lambdas(s: SparseSmoothSignal, lambda_min: float, lambda_max: float, nb: int, L: float, theta: float,
                 operator_l2: None | str | LinearOperator = "Laplacian", psnr: float = 50.):
    s.H = get_MyMatrixFreeOperator(s.dim, L)
    s.gaussian_noise(psnr)
    op_l2 = get_L2_operator(s.dim, operator_l2)

    loss_x1 = []
    loss_x2 = []
    picks = []
    lambdas = np.linspace(lambda_min, lambda_max, nb)
    for l in lambdas:
        x1, x2 = test_best_lines(s, L, l, theta, psnr, op_l2)
        loss_x1.append(Wasserstein_distance(s.sparse, x1))
        loss_x2.append(nmse(s.smooth, x2))
        picks.append(picks_found(s.sparse, x1, 1))

    name = f"θ:{theta:.2f}, {L:.1%} measurements, PSNR:{psnr:.0f}, L2 operator:{operator_l2.__str__()}"
    print(f"Best value L1: {lambdas[np.argmin(loss_x1)]}")
    print(f"Best value L2: {lambdas[np.argmin(loss_x2)]}")
    plot_loss(lambdas, loss_x1, loss_x2, name, "λ")
    picks = np.array(picks)
    plot_picks(lambdas, len(np.argwhere(s.sparse >= 2)), picks[:, 0], picks[:, 1], "λ")


def test_noise(s: SparseSmoothSignal, psnr_min: float, psnr_max: float, nb: int, L: float, lambda_: float, theta: float,
               operator_l2: None | str | LinearOperator = "Laplacian"):
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

    name = f"λ:{lambda_:.2f}, θ:{theta:.2f}, {L:.1%} measurements, L2 operator:{operator_l2.__str__()}"
    print(f"Best value L1: {psnrs[np.argmin(loss_x1)]}")
    print(f"Best value L2: {psnrs[np.argmin(loss_x2)]}")
    plot_loss(psnrs, loss_x1, loss_x2, name, "PSNR")


def get_best_lines(s: SparseSmoothSignal, L: float):
    s.H = MyMatrixFreeOperator(s.dim)
    y = np.abs(s.y)
    y[0] = 0
    return np.sort(np.argsort(y.ravel())[-int(L * d[0] * d[1]):])


def test_best_lines(s: SparseSmoothSignal, L: float, lambda_: float, theta: float, psnr: float,
                    operator_l2: None | str | LinearOperator = "Laplacian"):
    s.psnr = psnr
    best_lines = get_best_lines(s, L)
    s.H = MyMatrixFreeOperator(s.dim, best_lines)
    x1, x2 = test(s, lambda_ * theta, lambda_ * (1 - theta), get_L2_operator(d, operator_l2))
    return x1, x2


def picks_found(original_sparse: np.ndarray, reconstructed_sparse: np.ndarray, threshold: float = 0.75):
    sp1 = original_sparse.ravel()
    sp2 = reconstructed_sparse.ravel()
    picks = np.argwhere(sp1 >= 2)
    found = np.sum(sp2[picks] > threshold)
    wrong_pick = np.sum(sp2 > threshold) - found
    print(f"Picks in the original image : {len(picks)}")
    print(f"Picks found : {found}")
    print(f"Wrong picks found : {wrong_pick}")
    return found, wrong_pick


def picks_intensity(original_sparse: np.ndarray, reconstructed_sparse: np.ndarray):
    sp1 = original_sparse.ravel()
    sp2 = reconstructed_sparse.ravel()
    picks = np.argwhere(sp1 >= 2)
    intensity = sp2[picks] / sp1[picks]
    print(f"Mean intensity of the reconstructed picks : {np.mean(intensity):.1%}")


def compare_smoothing_operator(s: SparseSmoothSignal):
    s1.H = MyMatrixFreeOperator(d, get_best_lines(s1, L))
    lambda_ = 0.2
    theta = 0.5

    _, s_None = test(s, lambda_ * theta, 2 * lambda_ * (1 - theta), None)

    lambda_ = 0.2
    theta = 0.15
    _, s_G = test(s, lambda_ * theta, lambda_ * (1 - theta), get_L2_operator(d, "Gradient"))
    _, s_L = test(s, lambda_ * theta, lambda_ * (1 - theta), get_L2_operator(d, "Laplacian"))

    plot_4_smooth(s.smooth, s_None, s_G, s_L)


def compare_smooth(x_smooth: np.ndarray, x_tik: np.ndarray, x_tik_op: np.ndarray, x_lasso: np.ndarray, name: str = ""):
    plot_solvers(x_smooth, x_tik, x_tik_op, x_lasso, name, max_range=1)


def compare_sparse(x_sparse: np.ndarray, x_tik: np.ndarray, x_tik_op: np.ndarray, x_lasso: np.ndarray, name: str = ""):
    plot_solvers(x_sparse, x_tik, x_tik_op, x_lasso, name, min_range=1)


def compare_choose_of_lines(s: SparseSmoothSignal, L: float, lambda_: float, theta: float, psnr: float,
                            operator_l2: None | str | LinearOperator = "Laplacian"):
    s.psnr = psnr
    x1_best, x2_best = test_best_lines(s, L, lambda_, theta, psnr, operator_l2)
    s.H = MyMatrixFreeOperator(s.dim, int(L * s.dim[0] * s.dim[1]))
    x1_random, x2_random = test(s, lambda_ * theta, lambda_ * (1 - theta), get_L2_operator(s.dim, operator_l2))
    s.H = get_MyMatrixFreeOperator(s.dim, L)
    x1_gaus, x2_gaus = test(s, lambda_ * theta, lambda_ * (1 - theta), get_L2_operator(s.dim, operator_l2))

    name = f"λ:{l:.2f}, θ:{t:.2f}, {L:.1%} measurements, PSNR:{psnr:.0f}, L2 operator:{operator_l2.__str__()}"
    plot_8(s.sparse, s.smooth, x1_best, x2_best, x1_random, x2_random, x1_gaus, x2_gaus, name)


if __name__ == '__main__':
    d = (128, 128)
    seed = 11
    s1 = SparseSmoothSignal(d)
    s1.random_sparse(seed)
    s1.random_smooth(seed)
    L = 0.1
    l = 0.2
    t = 0.15
    psnr = 50.
    s1.psnr = psnr
    s1.H = MyMatrixFreeOperator(d, get_best_lines(s1, L))

    # compare_choose_of_lines(s1, L, l, t, psnr, "Gradient")
    # compare_smoothing_operator(s1)
    # x1, x2 = test_best_lines(s1, L, l, t, psnr, "Gradient")
    # name = f"λ:{l:.2f}, θ:{t:.2f}, {L:.1%} measurements, PSNR:{psnr:.0f}, L2 operator: Gradient"
    # plot_4(s1.sparse, s1.smooth, x1, x2, name)
    # picks_found(s1.sparse, x1, 2)
    # picks_intensity(s1.sparse, x1)
    # test_numbers_of_measurements(s1, 0.1, 0.75, 25, 0.1, 0.1, "Laplacian", 40.)
    # test_thetas(s1, 0.05, 0.95, 50, L, l, "Gradient", psnr)
    # test_lambdas(s1, 0.05, 2, 50, L, t, "Gradient", psnr)
    # test_noise(s1, 0., 50., 25, 0.25, 0.1, 0.1, "Laplacian")

    # name = f"λ:{l:.2f}, θ:{t:.2f}, {L:.1%} measurements, PSNR:{psnr:.0f}, L2 operator: Laplacian"
    #
    # x_sp, x_sm, x_t, x_t_op, x_l = test_solvers(s1, l * t, l * (1-t), "Laplacian", name)
    # compare_smooth(x_sm, x_t, x_t_op, x_l)
    # compare_sparse(x_sp, x_t, x_t_op, x_l)

    # test_hyperparameters(s1, [0.1], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3], ["Laplacian"], [20.0])

    plt.show()
