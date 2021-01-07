from koopman_probabilistic import *
import numpy as np
import matplotlib.pyplot as plt


def koopman_main():
    prediction_name = None
    # prediction_name = "sin17_koopman.npy"
    # x = np.sin(2*np.pi/24*np.arange(5000))**17
    # x = generate_frac_seq(5000, 0.81, n=8)
    # x = np.expand_dims(x,-1).astype(np.float32)

    mu_vec = 5 * np.sin(2 * np.pi / 24 * np.arange(5000))
    sigma_vec = np.sin(2 * np.pi / 24 * np.arange(5000) + 1.5) + 1.5
    rng = np.random.default_rng(425)
    x = rng.normal(mu_vec, sigma_vec).astype(np.float32)
    x = np.expand_dims(x, 1)

    mu_file = "mu_separated.npy"
    sigma_file = "sigma_separated.npy"
    try:
        raise IOError
        mu_hat = np.load(mu_file)
        sigma_hat = np.load(sigma_file)
    except IOError or FileNotFoundError as e:
        print(e)
        k = KoopmanProb(FullyConnectedNLL(x_dim=1, num_freqs_mu=1, num_freqs_sigma=1, n=512), device='cpu')
        k.fit(x[:3500], iterations=100, interval=25, verbose=True)
        mu_hat, sigma_hat = k.predict(5000)
        np.save(mu_file, mu_hat)
        np.save(sigma_file, sigma_hat)

    slc = -100
    plt.plot(x[slc:], label="data")
    plt.plot(mu_hat[slc:, 0], label="koopman")
    plt.plot(mu_hat[slc:, 0] + 2 * sigma_hat[slc:, 0], "--", color="black", label="koopman 95% CI")
    plt.plot(mu_hat[slc:, 0] - 2 * sigma_hat[slc:, 0], "--", color="black")
    plt.legend()
    plt.show()


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
    # fourier_main()