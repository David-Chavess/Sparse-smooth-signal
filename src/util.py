from __future__ import annotations

from typing import Tuple, List

import numpy as np
from matplotlib import pyplot as plt
from pycsou.core import LinearOperator
from pycsou.linop import Gradient, Laplacian
from scipy.stats import wasserstein_distance

from src import SparseSmoothSignal
from src.solver import MyMatrixFreeOperator


def nmse(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Compute the normalized mean square error between x1 and x2 normalized by the power of x1

    Parameters
    ----------
    x1 : np.ndarray
        Signal x1 used as a normalizer
    x2 : np.ndarray
        Signal x2

    Returns
    -------
    float
       NMSE between x1 and x2
    """
    return np.mean((x1 - x2) ** 2) / np.mean(x1 ** 2)


def wasserstein_dist(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Compute the wasserstein distance between x1 and x2

    Parameters
    ----------
    x1 : np.ndarray
        Signal x1
    x2 : np.ndarray
        Signal x2

    Returns
    -------
    float
        Wasserstein distance
    """
    return wasserstein_distance(x1.ravel() / np.sum(x1), x2.ravel() / np.sum(x2))


def get_L2_operator(dim: Tuple[int, int], op_l2: None | str | LinearOperator) -> LinearOperator:
    """
    Gives the operator used for the L2 penalty term in the solver.
    Used to get frequently used operator by their name

    Parameters
    ----------
    dim : Tuple[int, int]
        Dimension of the image
    op_l2 : None | str | LinearOperator
        Name of the operator we want to get.
        If None or LinearOperator it does nothing

    Returns
    -------
    LinearOperator
        The corresponding operator

    """
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


def random_points(dim: Tuple[int, int], size: int) -> np.ndarray:
    """
    Generates random 2d discrete point from a gaussian

    Parameters
    ----------
    dim : Tuple[int, int]
        Dimension of the image
    size : int
        Number of points

    Returns
    -------
    np.ndarray
        Array of 2d points
    """
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

    return np.array(list(points))


def random_low_freq_lines(dim: Tuple[int, int], size: int) -> np.ndarray:
    """
    Generates a list of lines index from random_points()

    Parameters
    ----------
    dim : Tuple[int, int]
        Dimension of the image
    size : int
        Number of lines

    Returns
    -------
    np.ndarray
        Lines to keep as an array of index
    """
    points = random_points(dim, size)
    index = points[:, 0] + points[:, 1] * dim[0]
    return index


def get_low_freq_operator(dim: Tuple[int, int], L: float) -> MyMatrixFreeOperator:
    """
    Get a MyMatrixFreeOperator with more low frequency measurements

    Parameters
    ----------
    dim : Tuple[int, int]
        Dimension of the image
    L : float
        Number of measurements in percentage between 0 and 1

    Returns
    -------
    MyMatrixFreeOperator
        MyMatrixFreeOperator with L% of measurements
    """
    return MyMatrixFreeOperator(dim, random_low_freq_lines(dim, int(L * dim[0] * dim[1])))


def get_best_lines(s: SparseSmoothSignal, L: float) -> np.ndarray:
    """
    Generates a list of lines index

    Parameters
    ----------
    s : SparseSmoothSignal
        Simulated signal to reconstruct
    L : float
        Number of measurements in percentage between 0 and 1

    Returns
    -------
    np.ndarray
        Lines to keep as an array of index
    """
    y = np.abs(np.fft.fft2(s.x)).ravel()
    y[0] = 0
    return np.sort(np.argsort(y)[-int(L * s.dim[0] * s.dim[1]):])


def get_best_freq_operator(s: SparseSmoothSignal, L: float) -> MyMatrixFreeOperator:
    """
    Get a MyMatrixFreeOperator with the best frequency measurements, taken as the highest fourier coefficient of
    our simulated signal

    Parameters
    ----------
    s : SparseSmoothSignal
        Simulated signal
    L : float
        Number of measurements in percentage between 0 and 1

    Returns
    -------
    MyMatrixFreeOperator
        MyMatrixFreeOperator with L% of measurements
    """
    return MyMatrixFreeOperator(s.dim, get_best_lines(s, L))


def plot_solvers(x: np.ndarray, x_tik: np.ndarray, x_tik_op: np.ndarray, x_lasso: np.ndarray, name: str = "",
                 min_range: float = 0,
                 max_range: float = SparseSmoothSignal.MAX_SMOOTH_AMPLITUDE + SparseSmoothSignal.MAX_SPARSE_AMPLITUDE) -> None:
    """
    Plot the output images of the different solvers

    Parameters
    ----------
    x : np.ndarray
        Image from Sparse + Smooth solver
    x_tik : np.ndarray
        Image from Tikhonov solver
    x_tik_op : np.ndarray
        Image from Tikhonov solver with an operator
    x_lasso : np.ndarray
        Image from Lasso solver
    name : str
        Name used as title, usually contains the parameters used in the solver
    min_range : float
        Minimum value of the plot
    max_range : float
        Maximum value of the plot
    """
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


def plot_reconstruction(x_sparse: np.ndarray, x_smooth: np.ndarray, x_reconst_sparse: np.ndarray,
                        x_reconst_smooth: np.ndarray, name: str = "") -> None:
    """
    Plot the original image signal and the reconstructed one

    Parameters
    ----------
    x_sparse : np.ndarray
        Original sparse image
    x_smooth : np.ndarray
        Original smooth image
    x_reconst_sparse : np.ndarray
        Reconstructed sparse image
    x_reconst_smooth : np.ndarray
        Reconstructed smooth image
    name : str
        Name used as title, usually contains the parameters used in the solver
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
    fig.canvas.manager.set_window_title(f'Spare + Smooth Signal : {name}')
    fig.suptitle(name)

    im_p = ax1.imshow(x_sparse, vmin=0, vmax=SparseSmoothSignal.MAX_SMOOTH_AMPLITUDE + SparseSmoothSignal.MAX_SPARSE_AMPLITUDE)
    ax1.axis('off')
    ax1.set_title("Original Sparse")

    im_s = ax2.imshow(x_smooth, vmin=0, vmax=SparseSmoothSignal.MAX_SMOOTH_AMPLITUDE)
    ax2.axis('off')
    ax2.set_title("Original Smooth")

    im = ax3.imshow(x_reconst_sparse, vmin=0, vmax=SparseSmoothSignal.MAX_SMOOTH_AMPLITUDE + SparseSmoothSignal.MAX_SPARSE_AMPLITUDE)
    ax3.axis('off')
    ax3.set_title("Reconstructed Sparse")

    im = ax4.imshow(x_reconst_smooth, vmin=0, vmax=SparseSmoothSignal.MAX_SMOOTH_AMPLITUDE)
    ax4.axis('off')
    ax4.set_title("Reconstructed Smooth")

    fig.subplots_adjust(bottom=0.05, top=0.9, left=0.01, right=0.92,
                        wspace=0.1, hspace=0.1)
    cb_ax = fig.add_axes([0.92, 0.05, 0.02, 0.85])
    cbar = fig.colorbar(im_s, cax=cb_ax)
    cb_ax = fig.add_axes([0.45, 0.05, 0.02, 0.85])
    cbar = fig.colorbar(im_p, cax=cb_ax)


def plot_reconstruction_measurements(x_sparse: np.ndarray, x_smooth: np.ndarray, x1_sparse: np.ndarray, x1_smooth: np.ndarray,
                                     x2_sparse: np.ndarray, x2_smooth: np.ndarray, x3_sparse: np.ndarray, x3_smooth: np.ndarray,
                                     name: str = "") -> None:
    """
    Plot the original image signal and 3 reconstruction made with different measurements

    Parameters
    ----------
    x_sparse : np.ndarray
        Original sparse image
    x_smooth : np.ndarray
        Original smooth image
    x1_sparse : np.ndarray
        Reconstructed sparse image with the best measurements
    x1_smooth : np.ndarray
        Reconstructed smooth image with the best measurements
    x2_sparse : np.ndarray
        Reconstructed sparse image with random uniform measurements
    x2_smooth : np.ndarray
        Reconstructed smooth image with random uniform measurements
    x3_sparse : np.ndarray
        Reconstructed sparse image with random_low_freq_lines()
    x3_smooth : np.ndarray
        Reconstructed smooth image with random_low_freq_lines()
    name : str
        Name used as title, usually contains the parameters used in the solver
    """
    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(20, 10))
    fig.canvas.manager.set_window_title(f'Spare + Smooth Signal : {name}')
    fig.suptitle(name)

    im_p = ax1.imshow(x_sparse, vmin=0, vmax=SparseSmoothSignal.MAX_SMOOTH_AMPLITUDE + SparseSmoothSignal.MAX_SPARSE_AMPLITUDE)
    ax1.axis('off')
    ax1.set_title("Original")

    im = ax2.imshow(x1_sparse, vmin=0, vmax=SparseSmoothSignal.MAX_SMOOTH_AMPLITUDE + SparseSmoothSignal.MAX_SPARSE_AMPLITUDE)
    ax2.axis('off')
    ax2.set_title("Best choose")

    im = ax3.imshow(x2_sparse, vmin=0, vmax=SparseSmoothSignal.MAX_SMOOTH_AMPLITUDE + SparseSmoothSignal.MAX_SPARSE_AMPLITUDE)
    ax3.axis('off')
    ax3.set_title("Random")

    im = ax4.imshow(x3_sparse, vmin=0, vmax=SparseSmoothSignal.MAX_SMOOTH_AMPLITUDE + SparseSmoothSignal.MAX_SPARSE_AMPLITUDE)
    ax4.axis('off')
    ax4.set_title("Random with more low frequency")

    im_s = ax5.imshow(x_smooth, vmin=0, vmax=SparseSmoothSignal.MAX_SMOOTH_AMPLITUDE)
    ax5.axis('off')

    im = ax6.imshow(x1_smooth, vmin=0, vmax=SparseSmoothSignal.MAX_SMOOTH_AMPLITUDE)
    ax6.axis('off')

    im = ax7.imshow(x2_smooth, vmin=0, vmax=SparseSmoothSignal.MAX_SMOOTH_AMPLITUDE)
    ax7.axis('off')

    im = ax8.imshow(x3_smooth, vmin=0, vmax=SparseSmoothSignal.MAX_SMOOTH_AMPLITUDE)
    ax8.axis('off')

    fig.subplots_adjust(bottom=0.05, top=0.9, left=0.01, right=0.92,
                        wspace=0.1, hspace=0.1)
    cb_ax = fig.add_axes([0.93, 0.50, 0.02, 0.40])
    cbar = fig.colorbar(im_p, cax=cb_ax)
    cb_ax = fig.add_axes([0.93, 0.05, 0.02, 0.40])
    cbar = fig.colorbar(im_s, cax=cb_ax)


def plot_smooth(x_smooth: np.ndarray, x_None: np.ndarray, x_Gradian: np.ndarray, x_Laplacian: np.ndarray) -> None:
    """
    Plot the smooth component of the original image and 3 reconstruction done with different operator used in L2 penalty
    term, the Gradian, the Laplacian and no operator.

    Parameters
    ----------
    x_smooth : np.ndarray
        Original smooth image
    x_None : np.ndarray
        Reconstructed smooth image without L2 operator
    x_Gradian : np.ndarray
        Reconstructed smooth image with the Gradian
    x_Laplacian : np.ndarray
        Reconstructed smooth image with the Laplacian
    """
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))

    im = ax1.imshow(x_smooth, vmin=0, vmax=SparseSmoothSignal.MAX_SMOOTH_AMPLITUDE)
    ax1.axis('off')
    ax1.set_title("Original Smooth")

    im = ax2.imshow(x_None)
    ax2.axis('off')
    ax2.set_title("Smooth without operator")

    im = ax3.imshow(x_Gradian, vmin=0, vmax=SparseSmoothSignal.MAX_SMOOTH_AMPLITUDE)
    ax3.axis('off')
    ax3.set_title("Smooth with Gradient")

    im = ax4.imshow(x_Laplacian, vmin=0, vmax=SparseSmoothSignal.MAX_SMOOTH_AMPLITUDE)
    ax4.axis('off')
    ax4.set_title("Smooth with Laplacian")

    fig.subplots_adjust(bottom=0.05, top=0.9, left=0.01, right=0.92,
                        wspace=0.1, hspace=0.1)
    cb_ax = fig.add_axes([0.93, 0.05, 0.02, 0.85])
    cbar = fig.colorbar(im, cax=cb_ax)


def plot_loss(x: np.ndarray, loss_sparse: List[float], loss_smooth: List[float], name: str = "", var: str = ""):
    """
    Plot the loss between the original image and the reconstructed one for all value in x. The loss is done with the
    Wasserstein distance for the sparse component and NMSE for the smooth one.

    Parameters
    ----------
    x :  np.ndarray
        Points of the x axe at which we computed the loss
    loss_sparse :  List[float]
        Wasserstein distance of the sparse component
    loss_smooth :  List[float]
        NMSE of the smooth component
    name : str
        Name used as title, usually contains the parameters used in the solver
    var : str
        Name of the variable we test
    """
    fig, (ax1, ax2), = plt.subplots(1, 2, figsize=(10, 5))
    fig.canvas.manager.set_window_title(f'Loss comparison')
    fig.suptitle(name)

    ax1.plot(x, loss_sparse)
    ax1.set_xlabel(var)
    ax1.set_ylabel("Wasserstein distance")
    ax1.set_title("L1 loss")

    ax2.plot(x, loss_smooth)
    ax2.set_xlabel(var)
    ax2.set_ylabel("NMSE")
    ax2.set_title("L2 loss")


def plot_peaks(x: np.ndarray, nb_peaks: float, peak_found: np.ndarray, wrong_peaks_found: np.ndarray,
               threshold: float, var: str = "") -> None:
    """
    Plot the numbers of "peaks" we can recover in the sparse component at each value of x.

    Parameters
    ----------
    x : np.ndarray
        Points of the x axe
    nb_peaks : float
        Number of "peak" in the original sparse component
    peak_found : np.ndarray
        Number of "peak" found
    wrong_peaks_found : np.ndarray
        Number of "peak" found that are not in the original sparse component
    threshold :
        Threshold used to determine if there is a "peak"
    var : str
        Name of the variable we test
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    fig.canvas.manager.set_window_title(f'Recovered peak comparison')
    ax.set_title(f"Recovered peak comparison, Threshold = {threshold}")

    ax.semilogy(x, peak_found, label='Peaks found')
    ax.semilogy(x, wrong_peaks_found, label='Wrong Peaks found')
    ax.axhline(y=nb_peaks, label='number of peaks', linestyle='--')
    ax.set_xlabel(var)
    ax.set_ylabel("Number of peaks")
    ax.legend()


def peaks_found(original_sparse: np.ndarray, reconstructed_sparse: np.ndarray, threshold: float = 0.75) \
        -> (np.ndarray, np.ndarray):
    """
    Compute the number of "peaks" found in the reconstructed sparse that are in the original and the number of "wrong
    peak" found that are not in the original signal.

    Parameters
    ----------
    original_sparse : np.ndarray
        Original sparse component
    reconstructed_sparse : np.ndarray
        Reconstructed sparse component
    threshold : float
        Threshold used to determine if there is a "peak"

    Returns
    -------
    (np.ndarray, np.ndarray)
        The number of "peaks" and the number of "wrong peaks"
    """
    sp1 = original_sparse.ravel()
    sp2 = reconstructed_sparse.ravel()
    peaks = np.argwhere(sp1 >= 2)
    found = np.sum(sp2[peaks] > threshold)
    wrong_peak = np.sum(sp2 > threshold) - found
    return found, wrong_peak
