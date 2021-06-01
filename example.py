import numpy as np
import matplotlib.pyplot as plt

import koopman_probabilistic
import model_objs

# generate toy data with periodic uncertainty
t = np.linspace(0, 1000 * np.pi, 10000)
mu_t = np.cos(t) * np.sin(t) ** 2 - 0.5
sigma_t = 0.1 * (1 + 0.7 * np.sin(t)) ** 2
x = np.random.normal(mu_t, sigma_t).reshape(-1, 1)

# define a model
periods = [20,]  # 20 idxs is a period
model_obj = model_objs.SkewNLLwithTime(x_dim=x.shape[1], num_freqs=[len(periods),] * 3)

# train the model
k = koopman_probabilistic.KoopmanProb(model_obj, device='cpu', num_fourier_modes=len(periods))
k.find_fourier_omegas(x, hard_code=periods)
k.fit(x, iterations=100, verbose=True)
params = k.predict(T=15000)
loc_hat, scale_hat, alpha_hat = params
x_hat = model_obj.mean(params)
std_hat = model_obj.std(params)

# plot
plt.plot(x_hat, "tab:orange", label="$\hat x$")
plt.plot(x, "tab:blue", label="$x$")
plt.plot(x_hat + std_hat, "--k", label="$\hat x \pm \hat \sigma$")
plt.plot(x_hat - std_hat, "--k")
plt.xlim([9900, 10100])
plt.legend()
plt.show()

plt.plot(mu_t, label="$\mu$")
plt.plot(x_hat, ":k", label="$\hat \mu$")
plt.plot(sigma_t, label="$\sigma$")
plt.plot(std_hat, "--k", label="$\hat \sigma$")
plt.xlim([9900, 10100])
plt.legend()
plt.show()
