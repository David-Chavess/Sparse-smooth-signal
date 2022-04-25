from __future__ import annotations

from src.lasso_solver import LassoSolver
from src.sparse_smooth_solver import SparseSmoothSolver
from src.tikhonov_solver import TikhonovSolver
from src.util import *


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
    peaks_intensity(s.sparse, x_sparse)
    peaks_found(s.sparse, x_tik, 1)
    peaks_intensity(s.sparse, x_tik)
    peaks_found(s.sparse, x_tik_op, 1)
    peaks_intensity(s.sparse, x_tik_op)
    peaks_found(s.sparse, x_lasso, 1)
    peaks_intensity(s.sparse, x_lasso)
    return x_sparse, x_smooth, x_tik, x_tik_op, x_lasso


def test_hyperparameters(s: SparseSmoothSignal, L: List[float], lambdas: List[float], thetas: List[float],
                         operators_l2: List[None | str | LinearOperator], psnr: List[float]):
    loss_x = {}
    loss_x1 = {}
    loss_x2 = {}
    for op in operators_l2:
        op_l2 = get_L2_operator(s.dim, op)
        for h in L:
            s.H = get_low_freq_operator(s.dim, h)
            for p in psnr:
                s.gaussian_noise(p)
                for l in lambdas:
                    for t in thetas:
                        name = f"λ:{l:.2f}, θ:{t:.2f}, {h:.1%} measurements, PSNR:{p:.0f}, L2 operator:{op.__str__()}"
                        x1, x2 = test(s, l * t, l * (1 - t), op_l2)
                        plot_reconstruction(s.sparse, s.smooth, x1, x2, name)
                        loss_x[name] = nmse(s.x, x1 + x2)
                        loss_x1[name] = wasserstein_dist(s.sparse, x1)
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
        s.H = get_low_freq_operator(s.dim, l)
        x1, x2 = test(s, lambda_ * theta, lambda_ * (1 - theta), op_l2)
        loss_x1.append(wasserstein_dist(s.sparse, x1))
        loss_x2.append(nmse(s.smooth, x2))

    name = f"λ:{lambda_:.2f}, θ:{theta:.2f}, PSNR:{psnr:.0f}, L2 operator:{operator_l2.__str__()}"
    print(f"Best value L1: {measurements[np.argmin(loss_x1)]}")
    print(f"Best value L2: {measurements[np.argmin(loss_x2)]}")
    plot_loss(measurements * 100, loss_x1, loss_x2, name, "Numbers of measurements")


def test_thetas(s: SparseSmoothSignal, theta_min: float, theta_max: float, nb: int, L: float, lambda_: float,
                operator_l2: None | str | LinearOperator = "Laplacian", psnr: float = 50., threshold: float = 1.):
    s.H = get_low_freq_operator(s.dim, L)
    s.gaussian_noise(psnr)
    op_l2 = get_L2_operator(s.dim, operator_l2)

    loss_x1 = []
    loss_x2 = []
    peaks = []
    thetas = np.linspace(theta_min, theta_max, nb)
    for t in thetas:
        x1, x2 = test_best_lines(s, L, lambda_, t, psnr, op_l2)
        loss_x1.append(wasserstein_dist(s.sparse, x1))
        loss_x2.append(nmse(s.smooth, x2))
        peaks.append(peaks_found(s.sparse, x1, threshold))

    name = f"λ:{lambda_:.2f}, {L:.1%} measurements, PSNR:{psnr:.0f}, L2 operator:{operator_l2.__str__()}"
    print(f"Best value L1: {thetas[np.argmin(loss_x1)]}")
    print(f"Best value L2: {thetas[np.argmin(loss_x2)]}")
    plot_loss(thetas, loss_x1, loss_x2, name, "θ")
    peaks = np.array(peaks)
    plot_peaks(thetas, len(np.argwhere(s.sparse >= 2)), peaks[:, 0], peaks[:, 1], threshold, "θ")


def test_lambdas(s: SparseSmoothSignal, lambda_min: float, lambda_max: float, nb: int, L: float, theta: float,
                 operator_l2: None | str | LinearOperator = "Laplacian", psnr: float = 50., threshold: float = 1.):
    s.H = get_low_freq_operator(s.dim, L)
    s.gaussian_noise(psnr)
    op_l2 = get_L2_operator(s.dim, operator_l2)

    loss_x1 = []
    loss_x2 = []
    peaks = []
    lambdas = np.linspace(lambda_min, lambda_max, nb)
    for l in lambdas:
        x1, x2 = test_best_lines(s, L, l, theta, psnr, op_l2)
        loss_x1.append(wasserstein_dist(s.sparse, x1))
        loss_x2.append(nmse(s.smooth, x2))
        peaks.append(peaks_found(s.sparse, x1, 1))

    name = f"θ:{theta:.2f}, {L:.1%} measurements, PSNR:{psnr:.0f}, L2 operator:{operator_l2.__str__()}"
    print(f"Best value L1: {lambdas[np.argmin(loss_x1)]}")
    print(f"Best value L2: {lambdas[np.argmin(loss_x2)]}")
    plot_loss(lambdas, loss_x1, loss_x2, name, "λ")
    peaks = np.array(peaks)
    plot_peaks(lambdas, len(np.argwhere(s.sparse >= 2)), peaks[:, 0], peaks[:, 1], threshold, "λ")


def test_noise(s: SparseSmoothSignal, psnr_min: float, psnr_max: float, nb: int, L: float, lambda_: float, theta: float,
               operator_l2: None | str | LinearOperator = "Laplacian"):
    s.H = get_low_freq_operator(s.dim, L)
    op_l2 = get_L2_operator(s.dim, operator_l2)

    loss_x1 = []
    loss_x2 = []
    psnrs = np.linspace(psnr_min, psnr_max, nb)
    for p in psnrs:
        s.gaussian_noise(p)
        x1, x2 = test(s, lambda_ * theta, lambda_ * (1 - theta), op_l2)
        loss_x1.append(wasserstein_dist(s.sparse, x1))
        loss_x2.append(nmse(s.smooth, x2))

    name = f"λ:{lambda_:.2f}, θ:{theta:.2f}, {L:.1%} measurements, L2 operator:{operator_l2.__str__()}"
    print(f"Best value L1: {psnrs[np.argmin(loss_x1)]}")
    print(f"Best value L2: {psnrs[np.argmin(loss_x2)]}")
    plot_loss(psnrs, loss_x1, loss_x2, name, "PSNR")


def test_best_lines(s: SparseSmoothSignal, L: float, lambda_: float, theta: float, psnr: float,
                    operator_l2: None | str | LinearOperator = "Laplacian"):
    s.psnr = psnr
    best_lines = get_best_lines(s, L)
    s.H = MyMatrixFreeOperator(s.dim, best_lines)
    x1, x2 = test(s, lambda_ * theta, lambda_ * (1 - theta), get_L2_operator(d, operator_l2))
    return x1, x2


def compare_smoothing_operator(s: SparseSmoothSignal):
    s1.H = MyMatrixFreeOperator(d, get_best_lines(s1, L))
    lambda_ = 0.2
    theta = 0.5

    _, s_None = test(s, lambda_ * theta, 2 * lambda_ * (1 - theta), None)

    lambda_ = 0.2
    theta = 0.15
    _, s_G = test(s, lambda_ * theta, lambda_ * (1 - theta), get_L2_operator(d, "Gradient"))
    _, s_L = test(s, lambda_ * theta, lambda_ * (1 - theta), get_L2_operator(d, "Laplacian"))

    plot_smooth(s.smooth, s_None, s_G, s_L)


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
    s.H = get_low_freq_operator(s.dim, L)
    x1_gaus, x2_gaus = test(s, lambda_ * theta, lambda_ * (1 - theta), get_L2_operator(s.dim, operator_l2))

    name = f"λ:{l:.2f}, θ:{t:.2f}, {L:.1%} measurements, PSNR:{psnr:.0f}, L2 operator:{operator_l2.__str__()}"
    plot_3_reconstruction(s.sparse, s.smooth, x1_best, x2_best, x1_random, x2_random, x1_gaus, x2_gaus, name)


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
    s1.H = MyMatrixFreeOperator(d)

    # compare_choose_of_lines(s1, L, l, t, psnr, "Gradient")
    # compare_smoothing_operator(s1)
    # x1, x2 = test_best_lines(s1, L, l, t, psnr, "Gradient")
    # name = f"λ:{l:.2f}, θ:{t:.2f}, {L:.1%} measurements, PSNR:{psnr:.0f}, L2 operator: Laplacian"
    # plot_4(s1.sparse, s1.smooth, x1, x2, name)
    # peaks_found(s1.sparse, x1, 1)
    # peaks_intensity(s1.sparse, x1)
    # test_numbers_of_measurements(s1, 0.1, 0.75, 25, 0.1, 0.1, "Laplacian", 40.)
    # test_thetas(s1, 0.05, 0.95, 50, L, l, "Laplacian", psnr)
    # test_lambdas(s1, 0.05, 2, 50, L, t, "Laplacian", psnr)
    # test_noise(s1, 0., 50., 25, 0.25, 0.1, 0.1, "Laplacian")

    # name = f"λ:{l:.2f}, θ:{t:.2f}, {L:.1%} measurements, PSNR:{psnr:.0f}, L2 operator: Laplacian"
    #
    # x_sp, x_sm, x_t, x_t_op, x_l = test_solvers(s1, l * t, l * (1-t), "Laplacian", name)
    # compare_smooth(x_sm, x_t, x_t_op, x_l)
    # compare_sparse(x_sp, x_t, x_t_op, x_l)

    # test_hyperparameters(s1, [0.1], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3], ["Laplacian"], [20.0])

    plt.show()
