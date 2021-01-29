from koopman_probabilistic import *
import numpy as np
import matplotlib.pyplot as plt


def koopman_main():
    # seed = np.random.randint(1000)
    # np.random.seed(seed)
    # print("SEED:", seed)

    # mu_vec = 2 * np.sin(2 * np.pi / 24 * np.arange(5000))
    # sigma_vec = 3 * np.sin(2 * np.pi / 34 * np.arange(5000) + 1.5) + 3
    # rng = np.random.default_rng(seed)
    # x = rng.normal(mu_vec, sigma_vec).astype(np.float32)
    # x = np.expand_dims(x, 1)
    predict_through = 24000
    enrgy = np.load("energy_data.npy")

    mu_file = "3mu_energy.npy"
    sigma_file = "3sigma_energy.npy"
    params_file = "3mu8sigma_params.npy"
    try:
        raise IOError
        mu_hat = np.load(mu_file)
        sigma_hat = np.load(sigma_file)
    except IOError or FileNotFoundError as e:
        print(e)
        model = FullyConnectedNLL(x_dim=1, num_freqs_mu=2, num_freqs_sigma=2, n=512)
        k = KoopmanProb(model, device='cpu', sample_num=24, min_periods=2, num_fourier_modes=0)
        xt = enrgy[:5000]
        k.find_fourier_omegas(xt)
        k.fit(xt, iterations=150, interval=20, verbose=True, cutoff=99)  # slice must be at least 1000
        mu_hat, sigma_hat = k.predict(predict_through)
        np.save(mu_file, mu_hat)
        np.save(sigma_file, sigma_hat)
        np.save(params_file, np.array(list(k.parameters())))

    # print("SEED:", seed)
    slc = -2400
    # plt.scatter(np.arange(-slc), x[slc:], label="data")
    plt.plot(enrgy[:predict_through], label="data")
    plt.plot(mu_hat, label="koopman")
    plt.plot(mu_hat + sigma_hat, "--", color="black", label="koopman 68% CI")
    plt.plot(mu_hat - sigma_hat, "--", color="black")

    # plt.plot(mu_vec[slc:], label="real mu")
    # plt.plot(mu_hat[slc:, 0], label="koopman mu")
    # plt.legend()
    # plt.show()
    #
    # plt.plot(sigma_vec[slc:], label="real sigma")
    plt.plot(sigma_hat, label="koopman sigma")
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
