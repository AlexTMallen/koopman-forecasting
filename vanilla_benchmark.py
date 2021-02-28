#!/usr/bin/env python
# coding: utf-8

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


def pinball_loss(data, quant_preds, quantiles):
    loss = 0
    for t in range(len(data)):
        for i in range(len(quantiles)):
            q = quantiles[i]
            quant_pred = quant_preds[t][i]
            if quant_pred > data[t]:
                loss += (quant_pred - data[t]) * (1 - q)
            else:
                loss += (data[t] - quant_pred) * q
    return loss / (len(data) * len(quantiles))


def test(alldata, zone_name, plot=False):
    zone = alldata[zone_name]
    # 1/1/2005 (first data point) was Saturday, so starts with 5 (Monday is 0)
    one_week = [5, 6, 0, 1, 2, 3, 4]
    days_of_week = np.array(list(one_week[(i // 24) % 7] for i in range(len(zone["DEMAND"]))), dtype=np.uint8)
    slc = 361 * 24
    cap = 365 * 24
    demand = np.array(zone["DEMAND"][-(slc + cap):-cap], dtype=np.float64)
    hour = np.array(zone["Hour"][-(slc + cap):-cap], dtype=np.uint8)
    week = days_of_week[-(slc + cap):-cap]
    month = np.array(list(map(lambda x: int(x[5:7]), zone["Date"][-(slc + cap):-cap])), dtype=np.uint8)
    trend = np.arange(slc)
    weekhour = week * hour
    temp = np.array(zone["DryBulb"][-(slc + cap):-cap], dtype=np.float64)
    temp2 = temp ** 2
    temp3 = temp ** 3
    tempmonth = temp * month
    temp2month = temp2 * month
    temp3month = temp3 * month
    temphour = temp * hour
    temp2hour = temp2 * hour
    temp3hour = temp3 * hour

    reg = linear_model.LinearRegression()

    variables = [trend, month, week, hour, weekhour, temp, temp2, temp3, tempmonth, temp2month, temp3month,
                 temphour, temp2hour, temp3hour]
    variables = list(map(list, zip(*variables)))  # transpose
    reg.fit(variables, demand)

    if plot:
        fit = reg.predict(variables)
        plt.figure()
        plt.plot(demand)
        plt.plot(fit)
        plt.show()


    #  ###---TEST---###
    slc = 31 * 24
    gap = 52 * 24
    cap = 365 * 24 - slc - gap
    test_demand = np.array(zone["DEMAND"][-(slc + cap):-cap], dtype=np.float64)
    hour = np.array(zone["Hour"][-(slc + cap):-cap], dtype=np.uint8)
    week = week = days_of_week[-(slc + cap):-cap]
    month = np.array(list(map(lambda x: int(x[5:7]), zone["Date"][-(slc + cap):-cap])), dtype=np.uint8)
    trend = np.arange(slc, slc + len(demand))

    all_temps = np.array(zone["DryBulb"], dtype=np.float64)
    temps = []
    for year in range(10):
        for shift in range(9):
            offset = int(year * 365.24 + shift + 0.5) * 24
            temps.append(all_temps[offset + gap: offset + slc + gap])

    preds = []
    for temp in temps:
        weekhour = week * hour
        temp2 = temp ** 2
        temp3 = temp ** 3
        tempmonth = temp * month
        temp2month = temp2 * month
        temp3month = temp3 * month
        temphour = temp * hour
        temp2hour = temp2 * hour
        temp3hour = temp3 * hour
        test_variables = [trend, month, week, hour, weekhour, temp, temp2, temp3, tempmonth, temp2month, temp3month,
                          temphour, temp2hour, temp3hour]
        test_variables = list(map(list, zip(*test_variables)))  # transpose
        preds.append(reg.predict(test_variables))

    preds = np.array(preds)

    if plot:
        pred = np.mean(preds, axis=0)
        std = np.std(preds, axis=0)
        plt.figure()
        plt.plot(test_demand)
        plt.plot(pred)
        plt.plot(std + pred, linestyle="--", color="black", linewidth=0.5)
        plt.plot(pred - std, linestyle="--", color="black", linewidth=0.5)
        plt.show()

    quant_preds = []
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for t in range(preds.shape[1]):
        sorted_preds = np.sort(preds[:, t])
        quant_pred = []
        for q in quantiles:
            idx = int(q * preds.shape[0] + 0.5)
            quant_pred.append(sorted_preds[idx])
        quant_preds.append(quant_pred)

    return pinball_loss(test_demand, quant_preds, quantiles)


def get_losses(plot=False):
    with open("GEFCom2017//GEFCom2017-Qual//GEFCom2017Qual2005-2015.json", "r") as f:
        alldata = json.loads(f.read())

    losses = dict()
    for zone_name in alldata.keys():
        losses[zone_name] = test(alldata, zone_name, plot=plot)

    return losses


if __name__ == "__main__":
    losses = get_losses(plot=True)
    print(losses)
