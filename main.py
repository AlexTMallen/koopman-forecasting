import random
import torch
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import json
from datetime import datetime
import pandas as pd

seed = 633
print("[ Using Seed : ", seed, " ]")
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
numpy.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from koopman_probabilistic import *
from gefcom_koopman import *
from model_objs import *


def koopman_main():
    # seed = np.random.randint(1000)
    # np.random.seed(seed)
    # print("SEED:", seed)

    zone_name = "NH"

    with open("GEFCom2017//GEFCom2017-Qual//GEFCom2017QualAll.json") as f:
        all_data = json.loads(f.read())

    dates = np.array(list(map(pd.Timestamp, all_data["ISONE CA"]["Date"])))
    zones = list(all_data.keys())
    month_name = "May"  # <<<< CHOOSE A MONTH
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_idx = months.index(month_name)
    print("Testing on month", months[month_idx])
    test_start_date = pd.Timestamp(f"2017-{month_idx + 1}-01 00:00:00")
    test_start = np.argwhere(dates == test_start_date)[0, 0]
    test_length = 31 * 24

    delay_delta = pd.Timedelta(days=52)
    delay = delay_delta.days * 24
    train_end_date = test_start_date - delay_delta
    train_start_date = train_end_date.replace(year=train_end_date.year - 11)
    train_start = np.argwhere(dates == train_start_date)[0, 0]
    train_length_delta = train_end_date - train_start_date
    train_through = train_length_delta.days * 24
    pre_length_delta = train_length_delta * 0.5  # the length of time at the beginning of the trianing period to hide from mu
    pre_length = pre_length_delta.days * 24

    data = np.array([all_data[zone_name]["DEMAND"]], dtype=np.float64).T
    temps_original = np.array([all_data[zone_name]["DryBulb"]], dtype=np.float64).T
    x_original = data

    # rescale data
    mean = np.mean(data, axis=0)
    rnge = np.max(data, axis=0) - np.min(data, axis=0)
    data = (data - np.matlib.repmat(mean, data.shape[0], 1)) / np.matlib.repmat(rnge, data.shape[0], 1)
    print("data", data.shape)
    mean_tmp = np.mean(temps_original, axis=0)
    rnge_tmp = np.max(temps_original, axis=0) - np.min(temps_original, axis=0)
    temps = (temps_original - np.matlib.repmat(mean_tmp, temps_original.shape[0], 1)) / np.matlib.repmat(rnge_tmp, temps_original.shape[0], 1)
    print("temp data", temps.shape)

    # train_start = 0
    # train_through = (10 * 365 + 2) * 24
    predict_through = train_through + delay + test_length
    tt = temps[train_start:][:train_through]
    x = data[train_start:]
    x = x[:predict_through]
    xt = x[:train_through]
    mask = torch.ones(xt.shape, dtype=torch.uint8)
    mask[:pre_length] = 0

    now = ("_".join(str(datetime.now()).split())).replace(":", ".")
    data_name = "main_" + f"_train_start={train_start}_" + now

    num_freqs = [4, 4, 4]
    num_fourier = 4
    loss_weights = 1 + 0.4 * torch.cos(torch.linspace(0, 2 * np.pi, xt.shape[0]))
    mu_file = "forecasts//" + data_name + f"trained{train_start}-{train_through}_{num_freqs}mu.npy"
    sigma_file = "forecasts//" + data_name + f"trained{train_start}-{train_through}_{num_freqs}sigma.npy"
    alpha_file = "forecasts//" + data_name + f"trained{train_start}-{train_through}_{num_freqs}alpha.npy"
    print("x", x.shape)
    print("tt", tt.shape)
    print("xt", xt.shape)

    ### TRAIN ###
    model = GEFComSkewNLL(x_dim=xt.shape[1], num_freqs=num_freqs, n=254)
    k = GEFComKoopman(model, device='cpu', sample_num=24, num_fourier_modes=num_fourier, batch_size=32, loss_weights=loss_weights)

    k.find_fourier_omegas(xt, hard_code=[24, 168, 24 * 365.25 / 12, 24 * 365.25])

    # k.fit(xt, iterations=20, interval=10, verbose=False, cutoff=0, weight_decay=1e-10000, lr_theta=5e-4, lr_omega=0,
    #       num_slices=None)
    k.fit(xt, tt, iterations=30, interval=10, verbose=True, cutoff=0, weight_decay=1e-4, lr_theta=1e-4, lr_omega=0,
          training_mask=mask)

    ### FORECAST ###
    params = k.predict(predict_through, temps[train_start:][:predict_through])
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
