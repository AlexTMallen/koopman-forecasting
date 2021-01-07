from koopman_probabilistic import *
import numpy as np
import matplotlib.pyplot as plt


def koopman_main():
    seed = np.random.randint(1000)
    np.random.seed()
    print("SEED:", seed)

    mu_vec = 5 * np.sin(2 * np.pi / 24 * np.arange(5000))
    sigma_vec = np.sin(2 * np.pi / 24 * np.arange(5000) + 1.5) + 1.5
    rng = np.random.default_rng()
    x = rng.normal(mu_vec, sigma_vec).astype(np.float32)
    x = np.expand_dims(x, 1)

    mu_file = "mu.npy"
    sigma_file = "sigma.npy"
    try:
        mu_hat = np.load(mu_file)
        sigma_hat = np.load(sigma_file)
    except IOError or FileNotFoundError as e:
        print(e)
        k = KoopmanProb(FullyConnectedNLL(x_dim=1, num_freqs_mu=1, num_freqs_sigma=1, n=512), device='cpu')
        k.fit(x[:3500], iterations=40, interval=20, verbose=True)
        mu_hat, sigma_hat = k.predict(5000)
        np.save(mu_file, mu_hat)
        np.save(sigma_file, sigma_hat)

    print("SEED:", seed)
    slc = -100
    # plt.scatter(np.arange(-slc), x[slc:], label="data")
    plt.plot(x[slc:], label="data")
    plt.plot(mu_hat[slc:, 0], label="koopman")
    plt.plot(mu_hat[slc:, 0] + 2 * sigma_hat[slc:, 0], "--", color="black", label="koopman 95% CI")
    plt.plot(mu_hat[slc:, 0] - 2 * sigma_hat[slc:, 0], "--", color="black")
    plt.legend()
    plt.show()

    plt.plot(mu_vec[slc:], label="real mu")
    plt.plot(mu_hat[slc:, 0], label="koopman mu")
    plt.legend()
    plt.show()

    plt.plot(sigma_vec[slc:], label="real sigma")
    plt.plot(sigma_hat[slc:, 0], label="koopman sigma")
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
