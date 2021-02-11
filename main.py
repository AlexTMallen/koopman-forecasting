from koopman_probabilistic import *
import numpy as np
import matplotlib.pyplot as plt


def koopman_main():
    # seed = np.random.randint(1000)
    # np.random.seed(seed)
    # print("SEED:", seed)

    x = np.load("energy_data.npy")
    data_name = "energy_data"

    predict_through = x.shape[0]
    train_through = 2000
    xt = x[:train_through, :]

    num_freqs = [3, 3, 3]
    num_fourier = 2
    mu_file = "forecasts//" + data_name + f"trainedThrough{train_through}_{num_freqs}mu.npy"
    sigma_file = "forecasts//" + data_name + f"trainedThrough{train_through}_{num_freqs}sigma.npy"
    alpha_file = "forecasts//" + data_name + f"trainedThrough{train_through}_{num_freqs}alpha.npy"

    model = SkewNLL(x_dim=xt.shape[1], num_freqs=num_freqs, n=512)
    k = KoopmanProb(model, device='cpu', sample_num=24, num_fourier_modes=num_fourier)
    k.find_fourier_omegas(xt)

    k.fit(xt, iterations=30, interval=10, verbose=True, cutoff=61, weight_decay=1e-1000, lr_theta=5e-3, lr_omega=1e-8)
    mu_hat, sigma_hat, a_hat = k.predict(predict_through)
    np.save(mu_file, mu_hat)
    np.save(sigma_file, sigma_hat)
    np.save(alpha_file, a_hat)

    for dim in range(xt.shape[1]):
        plt.figure()
        # plt.scatter(np.arange(-slc), x[slc:], label="data")
        plt.plot(x[:predict_through, dim], label="data")
        plt.plot(mu_hat[:, dim], label="Koopman $\mu$", linewidth=0.8)
        plt.plot(mu_hat[:, dim] + sigma_hat[:, dim], "--", color="black", label="Koopman $\mu \pm \sigma$ ", linewidth=0.5)
        plt.plot(mu_hat[:, dim] - sigma_hat[:, dim], "--", color="black", linewidth=0.5)
        plt.plot(a_hat[:, dim], color="orange", linewidth=0.7, label="Koopman $\\alpha$")

        # plt.plot(mu_vec[slc:], label="real mu")
        # plt.plot(mu_hat[slc:, 0], label="koopman mu")
        # plt.legend()
        # plt.show()
        #
        # plt.plot(sigma_vec[slc:], label="real sigma")
        #     plt.plot(3*sigma_hat[:, dim], label="koopman $3\sigma$", linewidth=0.7)
        plt.title(f"{num_freqs}_trainedThrough{train_through}_" + data_name)
        plt.xlabel("t")
        plt.legend()
        plt.show()

    # plt.plot(mu_vec[slc:], label="real mu")
    # plt.plot(mu_hat[slc:, 0], label="koopman mu")
    # plt.legend()
    # plt.show()
    #
    # plt.plot(sigma_vec[slc:], label="real sigma")
    # plt.plot(sigma_hat, label="koopman sigma")
    # plt.legend()
    # plt.show()


def generate_frac_seq(num_iters, value, n=3):
    k = 2 + 2*np.cos(2 * np.math.pi / n)
    x_vals = []
    _generate_frac_period(n, value, x_vals, k)
    return np.tile(x_vals, num_iters // len(x_vals) + 1)


def _generate_frac_period(num_iters, value, x_vals, k):
    x_vals.append(value)
    if num_iters > 1:
        _generate_frac_period(num_iters - 1, k - k/value, x_vals, k)


if __name__ == "__main__":
    koopman_main()
