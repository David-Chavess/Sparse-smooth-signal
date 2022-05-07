from __future__ import annotations

from typing import Dict

import numpy as np

from src import SparseSmoothSignal
from src.lasso_solver import LassoSolver
from src.sparse_smooth_solver import SparseSmoothSolver
from src.tikhonov_solver import TikhonovSolver
from src.util import *


def off_set_smooth(smooth: np.ndarray, x_smooth: np.ndarray) -> np.ndarray:
    """
    Offset the smooth reconstruction because the reconstruction is 0-mean.

    Parameters
    ----------
    smooth : np.ndarray
        Smooth component of the original signal
    x_smooth : np.ndarray
        Smooth component to offset

    Returns
    -------
    np.ndarray
        The offset values
    """
    return np.mean(smooth) - np.mean(x_smooth)


def print_best(loss_x1: Dict, loss_x2: Dict) -> None:
    """
    Print the loss for each component of the reconstructed for different set of parameters.

    Parameters
    ----------
    loss_x1 : Dict
        Loss of the sparse component for each set of parameters
    loss_x2 : Dict
        Loss of the smooth component for each set of parameters
    """
    print("Sparse:")
    for x in sorted(loss_x1, key=loss_x1.get):
        print(f"{x} : {loss_x1[x]}")
    print("Smooth:")
    for x in sorted(loss_x2, key=loss_x2.get):
        print(f"{x} : {loss_x2[x]}")


def solve(s: SparseSmoothSignal, l1: float, l2: float, op: None | LinearOperator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the inverse problem for a simulated signal s.

    Parameters
    ----------
    s : SparseSmoothSignal
        Simulated signal to reconstruct
    l1 : float
        Weight of the L1 penalty
    l2 : float
        Weight of the L2 penalty
    op: None | LinearOperator
        Operator used in the L2 penalty

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple (x1, x2) with the sparse component x1 and the smooth component x2.
    """
    m = np.max(np.abs(s.H.adjoint(s.y)))
    solver = SparseSmoothSolver(s.y, s.H, l1 * m, l2 * m, op)
    x1, x2 = solver.solve()

    x1 = x1.reshape(s.dim)
    x2 = x2.reshape(s.dim)

    x1[x1 < 0] = 0
    # x2 += off_set_smooth(s.smooth, x2)

    return x1, x2


def solvers(s: SparseSmoothSignal, lambda1: float, lambda2: float,
            operator_l2: None | str | LinearOperator = "Laplacian",
            name: str = "") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the inverse problem for a simulated signal s with different solvers. It uses Lasso, Tikhonov with and without
    operator and our "Sparse + smooth" solver. Used to compare the performance of each one.

    Parameters
    ----------
    s : SparseSmoothSignal
        Simulated signal to reconstruct
    lambda1 : float
        Weight of the L1 penalty
    lambda2 : float
        Weight of the L2 penalty
    operator_l2 : None | str | LinearOperator
        Operator used in the L2 penalty
    name : str
        Name used as title, usually contains the parameters used in the solver

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Tuple (x_sparse, x_smooth, x_tik, x_tik_op, x_lasso) with the solutions of each solver
    """
    op = get_L2_operator(s.dim, operator_l2)

    x_sparse, x_smooth = solve(s, lambda1, lambda2, op)
    x_sparse = x_sparse.reshape(s.dim)
    x_smooth = x_smooth.reshape(s.dim)

    plot_reconstruction(s.sparse, s.smooth, x_sparse, x_smooth, name)

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

    peaks_found(s.sparse, x_sparse, 1)
    peaks_found(s.sparse, x_tik, 1)
    peaks_found(s.sparse, x_tik_op, 1)
    peaks_found(s.sparse, x_lasso, 1)
    return x_sparse, x_smooth, x_tik, x_tik_op, x_lasso


def test_hyperparameters(s: SparseSmoothSignal, L: float, lambdas1: List[float], lambdas2: List[float],
                         operators_l2: List[None | str | LinearOperator], psnr: List[float]) -> None:
    """
    Solve and plot all reconstruction made by the combination of all the parameters.

    Parameters
    ----------
    s : SparseSmoothSignal
        Simulated signal to reconstruct
    L : float
        Number of measurements in percentage between 0 and 1
    lambdas1 : List[float]
        Weights of the L1 penalty
    lambdas2 : List[float]
        Weights of the L1 penalty
    operators_l2 : List[None | str | LinearOperator]
        Operators used in the L2 penalty
    psnr : List[float]
        PSNRs of the measurements
    """
    loss_x1 = {}
    loss_x2 = {}
    for op in operators_l2:
        op_l2 = get_L2_operator(s.dim, op)
        for p in psnr:
            s.psnr = psnr
            for l1 in lambdas1:
                for l2 in lambdas2:
                    name = f"λ1:{l1:.2f}, λ2:{l2:.2f}, {L:.1%} measurements, PSNR:{p:.0f}, L2 operator:{op.__str__()}"
                    x1, x2 = solve(s, l1, l2, op_l2)
                    loss_x1[name] = wasserstein_dist(s.sparse, x1)
                    loss_x2[name] = nmse(s.smooth, x2)
                    plot_reconstruction(s.sparse, s.smooth, x1, x2, name)

    print_best(loss_x1, loss_x2)


def test_lambda1(s: SparseSmoothSignal, L: float, lambda1_min: float, lambda1_max: float, nb: int, lambda2: float | None = None, operator_l2: None | str | LinearOperator = "Laplacian", psnr: float = 50., threshold: float = 1.) -> None:
    """
    Test the lambda1 parameter of some fix parameters. It makes nb reconstructions and plot the loss of each
    component. It also plots the number of peaks recovered in the sparse component.

    Parameters
    ----------
    s : SparseSmoothSignal
        Simulated signal to reconstruct
    L : float
        Number of measurements in percentage between 0 and 1
    lambda1_min : float
        Minimum value of λ1
    lambda1_max : float
        Maximum value of λ1
    nb : int
        Number of reconstructions made
    lambda2 : float | None
        Weight of the L2 penalty, if None use lambda2 = 5 * lambda1
    operator_l2 : None | str | LinearOperator
        Operator used in the L2 penalty
    psnr : float
        PSNR of the measurements
    threshold : float
        Threshold used in plot_peaks to count the number of peaks recovered
    """
    s.psnr = psnr
    op_l2 = get_L2_operator(s.dim, operator_l2)

    loss_x1 = []
    loss_x2 = []
    nb_peaks_found = []

    # We sample a few peaks and a few non-peaks
    peaks = np.empty((nb, 25))
    non_peaks = np.empty((nb, 100))
    samples_peaks = np.argwhere(s.sparse.ravel() > SparseSmoothSignal.MIN_SPARSE_AMPLITUDE).ravel()
    samples_peaks = np.random.choice(samples_peaks, size=25, replace=False)
    samples_non_peaks = np.argwhere(s.sparse.ravel() < SparseSmoothSignal.MIN_SPARSE_AMPLITUDE).ravel()
    samples_non_peaks = np.random.choice(samples_non_peaks, size=100, replace=False)

    lambdas = np.linspace(lambda1_min, lambda1_max, nb)
    for i, l in enumerate(lambdas):
        if lambda2 is None:
            x1, x2 = solve(s, l, 5 * l, op_l2)
        else:
            x1, x2 = solve(s, l, lambda2, op_l2)
        loss_x1.append(wasserstein_dist(s.sparse, x1))
        loss_x2.append(nmse(s.smooth, x2))
        nb_peaks_found.append(peaks_found(s.sparse, x1, 1))
        peaks[i] = x1.ravel()[samples_peaks]
        non_peaks[i] = x1.ravel()[samples_non_peaks]

    if lambda2 is None:
        name = f"λ2: 5*λ1, {L:.1%} measurements, PSNR:{psnr:.0f}, L2 operator:{operator_l2.__str__()}"
    else:
        name = f"λ2:{lambda2:.2f}, {L:.1%} measurements, PSNR:{psnr:.0f}, L2 operator:{operator_l2.__str__()}"
    print(f"Best value for L1 penalty: {lambdas[np.argmin(loss_x1)]}")
    print(f"Best value for L2 penalty: {lambdas[np.argmin(loss_x2)]}")
    plot_loss(lambdas, loss_x1, loss_x2, name, "λ1")
    nb_peaks_found = np.array(nb_peaks_found)
    plot_peaks(lambdas, len(np.argwhere(s.sparse >= 2)), nb_peaks_found[:, 0], nb_peaks_found[:, 1], threshold, "λ1")

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    fig.canvas.manager.set_window_title("Following some pixel of the sparse component")
    ax.set_title("Intensity of some pixel of the sparse component")
    for i in range(non_peaks[0].size):
        line1, = ax.plot(lambdas, non_peaks[:, i], color='tab:orange')
    for i in range(peaks[0].size):
        line2, = ax.plot(lambdas, peaks[:, i], color='tab:blue')

    ax.set_ylabel("Pixel value")
    ax.set_xlabel("λ1")
    ax.legend([line1, line2], ["Zeros", "Peaks"])


def test_lambda2(s: SparseSmoothSignal, L: float, lambda2_min: float, lambda2_max: float, nb: int, lambda1: float | None = None, operator_l2: None | str | LinearOperator = "Laplacian", psnr: float = 50., threshold: float = 1.) -> None:
    """
    Test the lambda2 parameter of some fix parameters. It makes nb reconstructions and plot the loss of each
    component. It also plots the number of peaks recovered in the sparse component.

    Parameters
    ----------
    s : SparseSmoothSignal
        Simulated signal to reconstruct
    L : float
        Number of measurements in percentage between 0 and 1
    lambda2_min : float
        Minimum value of λ2
    lambda2_max : float
        Maximum value of λ2
    nb : int
        Number of reconstructions made
    lambda1 : float | None
        Weight of the L1 penalty, if None use lambda1 = 1/5 * lambda2
    operator_l2 : None | str | LinearOperator
        Operator used in the L2 penalty
    psnr : float
        PSNR of the measurements
    threshold : float
        Threshold used in plot_peaks to count the number of peaks recovered
    """
    s.psnr = psnr
    op_l2 = get_L2_operator(s.dim, operator_l2)

    loss_x1 = []
    loss_x2 = []
    peaks = []
    lambdas = np.linspace(lambda2_min, lambda2_max, nb)
    for l in lambdas:
        if lambda1 is None:
            x1, x2 = solve(s, l / 5, l, op_l2)
        else:
            x1, x2 = solve(s, lambda1, l, op_l2)
        loss_x1.append(wasserstein_dist(s.sparse, x1))
        loss_x2.append(nmse(s.smooth, x2))
        peaks.append(peaks_found(s.sparse, x1, threshold))

    if lambda1 is None:
        name = f"λ1: λ2/5, {L:.1%} measurements, PSNR:{psnr:.0f}, L2 operator:{operator_l2.__str__()}"
    else:
        name = f"λ1:{lambda1:.2f}, {L:.1%} measurements, PSNR:{psnr:.0f}, L2 operator:{operator_l2.__str__()}"
    print(f"Best value L1: {lambdas[np.argmin(loss_x1)]}")
    print(f"Best value L2: {lambdas[np.argmin(loss_x2)]}")
    plot_loss(lambdas, loss_x1, loss_x2, name, "λ2")
    peaks = np.array(peaks)
    plot_peaks(lambdas, len(np.argwhere(s.sparse >= 2)), peaks[:, 0], peaks[:, 1], threshold, "λ2")


def compare_smoothing_operator(s: SparseSmoothSignal) -> None:
    """
    Compare the reconstructions using the Gradient, the Laplacian and no operator in the L2 penalty with the original
    signal.

    Parameters
    ----------
    s : SparseSmoothSignal
        Simulated signal to reconstruct
    """
    lambda1 = 0.1
    lambda2 = 0.05

    _, s_None = solve(s, lambda1, lambda2, None)

    lambda1 = 0.02
    lambda2 = 0.05
    _, s_G = solve(s, lambda1, lambda2, get_L2_operator(s.dim, "Gradient"))
    _, s_L = solve(s, lambda1, lambda2, get_L2_operator(s.dim, "Laplacian"))

    plot_smooth(s.smooth, s_None, s_G, s_L)


def compare_measurements_methods(s: SparseSmoothSignal, L: float, lambda1: float, lambda2: float, psnr: float,
                                 operator_l2: None | str | LinearOperator = "Laplacian") -> None:
    """
    Compare the reconstructions using different choice of measurements with the original signal.

    Parameters
    ----------
    s : SparseSmoothSignal
        Simulated signal to reconstruct
    L : float
        Number of measurements in percentage between 0 and 1
    lambda1 : float
        Weight of the L1 penalty
    lambda2 : float
        Weight of the L2 penalty
    psnr : float
        PSNR of the measurements
    operator_l2 : None | str | LinearOperator
        Operator used in the L2 penalty
    """
    s.psnr = psnr
    l2_op = get_L2_operator(s.dim, operator_l2)

    H = s.H

    s.H = get_best_freq_operator(s, L)
    x1_best, x2_best = solve(s, lambda1, lambda2, l2_op)
    intensity = peaks_intensity(s.sparse, x1_best)
    print(f"Mean intensity of the reconstructed peaks (Highest Fourier coefficient) : {np.mean(intensity):.1%}")

    s.H = MyMatrixFreeOperator(s.dim, int(L * s.dim[0] * s.dim[1]))
    x1_random, x2_random = solve(s, lambda1, lambda2, l2_op)
    intensity = peaks_intensity(s.sparse, x1_random)
    print(f"Mean intensity of the reconstructed peaks (Random uniform) : {np.mean(intensity):.1%}")

    s.H = get_low_freq_operator(s.dim, L)
    x1_low, x2_low = solve(s, lambda1, lambda2, l2_op)
    intensity = peaks_intensity(s.sparse, x1_low)
    print(f"Mean intensity of the reconstructed peaks (Random Gaussian + Uniform) : {np.mean(intensity):.1%}")

    s.H = H

    name = f"λ1:{lambda1:.2f}, λ2:{lambda2:.2f}, {L:.1%} measurements, PSNR:{psnr:.0f}, L2 operator:{operator_l2.__str__()}"
    plot_reconstruction_measurements(s.sparse, s.smooth, x1_best, x2_best, x1_random, x2_random, x1_low, x2_low, name)
