from koopman_probabilistic import *
from model_objs import *
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import json
from datetime import datetime

def koopman_main():
    # seed = np.random.randint(1000)
    # np.random.seed(seed)
    # print("SEED:", seed)

    zone_name = "ISONE CA"

    with open("GEFCom2017//GEFCom2017-Qual//GEFCom2017Qual2005-2015.json") as f:
        all_data = json.loads(f.read())

    data = np.transpose(np.array([all_data[zone_name]["DEMAND"]], dtype=np.float64))

    # rescale data
    x_original = data
    mean = np.mean(data, axis=0)
    rnge = np.max(data, axis=0) - np.min(data, axis=0)
    data = (data - np.matlib.repmat(mean, data.shape[0], 1)) / np.matlib.repmat(rnge, data.shape[0], 1)
    print("data", data.shape)

    # train_start = 0
    # train_through = (10 * 365 + 2) * 24
    train_start = (9 * 365 + 2) * 24
    train_through = 365 * 24
    delay = 52 * 24
    predict_through = train_through + 24 * 31 + delay
    x = data[train_start:]
    x = x[:predict_through]
    xt = x[:train_through]

    now = ("_".join(str(datetime.now()).split())).replace(":", ".")
    data_name = "EXPnormality_" + f"_train_start={train_start}_" + now

    num_freqs = [5, 5, 5]
    num_fourier = 4
    loss_weights = 1 + 0.4 * torch.cos(torch.linspace(0, 2 * np.pi, xt.shape[0]))
    mu_file = "forecasts//" + data_name + f"trained{train_start}-{train_through}_{num_freqs}mu.npy"
    sigma_file = "forecasts//" + data_name + f"trained{train_start}-{train_through}_{num_freqs}sigma.npy"
    alpha_file = "forecasts//" + data_name + f"trained{train_start}-{train_through}_{num_freqs}alpha.npy"
    print("x", x.shape)
    print("xt", xt.shape)

    ### TRAIN ###
    model = AlternatingSkewNLL(x_dim=xt.shape[1], num_freqs=num_freqs, n=512)
    k = KoopmanProb(model, device='cpu', sample_num=24, num_fourier_modes=num_fourier, batch_size=32, loss_weights=loss_weights)

    k.find_fourier_omegas(xt, hard_code=[24, 168, 24 * 365.25 / 12, 24 * 365.25])

    # k.fit(xt, iterations=20, interval=10, verbose=False, cutoff=0, weight_decay=1e-10000, lr_theta=5e-4, lr_omega=0,
    #       num_slices=None)
    k.fit(xt, iterations=50, interval=10, verbose=True, cutoff=1, weight_decay=0, lr_theta=1e-4, lr_omega=0,
          num_slices=None)

    ### FORECAST ###
    params = k.predict(predict_through)
    mu_hat, sigma_hat, a_hat = params
    np.save(mu_file, mu_hat)
    np.save(sigma_file, sigma_hat)
    np.save(alpha_file, a_hat)

    mean_hat = model.mean(params)
    std_hat = model.std(params)

    dim = 0
    plt.figure()
    # plt.scatter(np.arange(-slc), x[slc:], label="data")
    plt.plot(x[:predict_through, dim], label="data")
    plt.plot(mean_hat[:, dim], label="Koopman mean", linewidth=1)
    plt.plot(mean_hat[:, dim] + std_hat[:, dim], "--", color="black", label="Koopman mean $\pm$ std", linewidth=0.5)
    plt.plot(mean_hat[:, dim] - std_hat[:, dim], "--", color="black", linewidth=0.5)
    # plt.plot(a_hat[:, dim], color="red", linewidth=0.3, label="Koopman $\\alpha$")
    # plt.plot(std_hat[:, dim], color="green", linewidth=0.7, label="Koopman std")

    plt.title(f"{num_freqs}_trainedThrough{train_through}_" + data_name)
    plt.xlabel("t")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    koopman_main()
