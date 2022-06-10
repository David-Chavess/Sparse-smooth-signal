import matplotlib

from src import *

if __name__ == '__main__':
    matplotlib.rcParams.update({'font.size': 14})
    d = (128, 128)
    seed = 42

    # s = SparseSmoothSignal(d)
    # s.random_sparse(seed)
    # s.random_smooth(seed)
    #
    # L = 0.1
    # l1 = 0.02
    # l2 = 0.06
    # psnr = 50.
    #
    # s.psnr = psnr
    # op_l2 = get_L2_operator(d, "Laplacian")
    # s.H = get_low_freq_operator(d, L)
    #
    # x_sparse, x_smooth, x_tik, x_tik_op, x_lasso = solvers(s, l1, l2, op_l2)
    # plot_solvers(x_sparse + x_smooth, x_tik, x_tik_op, x_lasso)
    # plot_solvers(x_sparse + x_smooth, x_tik, x_tik_op, x_lasso, max_range=SparseSmoothSignal.MAX_SMOOTH_AMPLITUDE)
    # test_hyperparameters(s, L, [0.02, 0.1, 0.2], [0.02, 0.1, 0.2], ["Laplacian"], [20., 50.])
    # test_tuning_parameters(s,  [0.02, 0.06, 0.15, 0.3], [0.02, 0.06, 0.15, 0.3], "Laplacian", 50.)
    # plot_sampling_methods(s, L)
    # compare_measurements_methods(s, L, l1, l2, psnr, "Laplacian")
    # compare_smoothing_operator(s)
    #
    # x1, x2 = solve(s, l1, l2, op_l2)
    # plot_reconstruction(s.sparse, s.smooth, x1, x2)
    # found, wrong = peaks_found(s.sparse, x1, 1)
    # print(len(np.argwhere(s.sparse >= 2)))
    # print(f"Peaks found : {found}")
    # print(f"Wrong peaks found : {wrong}")
    # intensity = peaks_intensity(s.sparse, x1)
    # print(f"Mean intensity of the reconstructed peaks : {np.mean(intensity):.1%}")
    #
    # test_lambda1(s, L, 0.01, 0.5, 50, 0.8, "Laplacian", psnr, 2)
    # test_lambda2(s, L, 0.1, 4, 50, 0.25, "Laplacian", psnr, 2)
    #
    # x1, x2 = solve(s, l1, l2, op_l2)
    # name = f"λ1: {l1:.2f}   λ2: {l2:.2f}"
    # plot_2_reconstruction(x1, x2, name)
    # plot_reconstruction(s.sparse, s.smooth, x1, x2)
    #
    # s.show()
    #
    # L = 0.1
    # l1 = 0.01
    # l2 = 0.2
    #
    # image = matplotlib.image.imread("images/image1.png")
    # x = np.mean(image, axis=-1)
    #
    # # s = SparseSmoothSignal(x.shape, x, np.zeros(x.shape), psnr=20.)
    # print(x.shape)
    #
    # figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
    # ax1.imshow(x, cmap="gray")
    # ax1.set_title("Original")
    #
    # op_l2 = get_L2_operator(x.shape, "Laplacian")
    # s.H = get_low_freq_operator(x.shape, L)
    # H = get_low_freq_operator(x.shape, L)
    # y = H @ x
    # # H = s.H
    # # y = s.y
    #
    # m = np.max(np.abs(H.adjoint(y)))
    # solver = SparseSmoothSolver(y, H, l1 * m, l2 * m, op_l2)
    # x1, x2 = solver.solve()
    #
    # x1 = x1.reshape(x.shape)
    # x2 = x2.reshape(x.shape)
    #
    # x1[x1 < 0] = 0
    # x2[x2 < 0] = 0
    #
    # ax3.imshow(1 - x1, cmap="gray")
    # ax4.imshow(x2, cmap="gray")
    # ax3.set_title("Sparse")
    # ax4.set_title("Smooth")
    #
    # ax2.imshow(x1 + x2, cmap="gray")
    # ax2.set_title("Reconstruction")
    #
    # plt.show()
