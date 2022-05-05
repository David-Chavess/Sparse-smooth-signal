from src import *

if __name__ == '__main__':
    d = (128, 128)
    seed = 111
    s = SparseSmoothSignal(d)
    s.random_sparse(seed)
    s.random_smooth(seed)
    L = 0.1
    l1 = 0.02
    l2 = 0.05
    psnr = 20.
    s.psnr = psnr
    op_l2 = get_L2_operator(d, "Laplacian")
    s.H = get_low_freq_operator(d, L)

    name = f"λ1:{l1:.2f}, λ2:{l2:.2f}, {L:.1%} measurements, PSNR:{psnr:.0f}, L2 operator:Laplacian"

    # plot_sampling_methods(s1, L)
    # compare_measurements_methods(s1, L, l1, l2, psnr, "Laplacian")
    # compare_smoothing_operator(s)

    x1, x2 = solve(s, l1, l2, op_l2)
    plot_reconstruction(s.sparse, s.smooth, x1, x2, name)
    found, wrong = peaks_found(s.sparse, x1, 0.75)
    print(len(np.argwhere(s.sparse >= 2)))
    print(f"Peaks found : {found}")
    print(f"Wrong peaks found : {wrong}")

    # test_lambda1(s1, L, 0.001, 0.1, 10, 0.2, "Laplacian", psnr, 1)

    s.show()
