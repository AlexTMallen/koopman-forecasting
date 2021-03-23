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


def test(all_demand, hours_of_day, all_temps, months_of_year, train_through=None, gap=0, test_through=None, plot=False):
    days_of_week = np.array(list((i // 24) % 7 for i in range(len(all_demand))), dtype=np.uint8)

    cap = train_through
    demand = all_demand[:cap]
    hour = np.zeros((23, cap), dtype=np.uint8)
    for i in range(1, 24):
        hour[i - 1] = (i == hours_of_day[:cap])
    week = np.zeros((6, cap), dtype=np.uint8)
    for i in range(6):
        week[i] = (i == days_of_week[:cap])
    month = np.zeros((11, cap), dtype=np.uint8)
    for i in range(1, 12):
        month[i - 1] = (i == months_of_year[:cap])
    trend = np.arange(cap)
    weekhour = np.zeros((6 * 23, cap))
    for i in range(6):
        for j in range(23):
            weekhour[23 * i + j] = week[i] * hour[j]
    temp = all_temps[-len(all_demand):][:cap]
    temp2 = temp ** 2
    temp3 = temp ** 3
    tempmonth = np.array(list(map(lambda x: temp * x, month)))
    temp2month = np.array(list(map(lambda x: temp2 * x, month)))
    temp3month = np.array(list(map(lambda x: temp3 * x, month)))
    temphour = np.array(list(map(lambda x: temp * x, hour)))
    temp2hour = np.array(list(map(lambda x: temp2 * x, hour)))
    temp3hour = np.array(list(map(lambda x: temp3 * x, hour)))

    reg = linear_model.LinearRegression()

    variables = [trend, *month, *week, *hour, *weekhour, temp, temp2, temp3, *tempmonth, *temp2month, *temp3month,
                 *temphour, *temp2hour, *temp3hour]
    variables = list(map(list, zip(*variables)))  # transpose
    reg.fit(variables, demand)

    if plot:
        fit = reg.predict(variables)
        plt.figure()
        plt.plot(demand)
        plt.plot(fit)
        plt.show()

    #  ###---TEST---###
    slc = test_through
    cap = train_through + slc + gap
    test_demand = all_demand[cap - slc:cap]
    hour = np.zeros((23, len(test_demand)), dtype=np.uint8)
    for i in range(1, 24):
        hour[i - 1] = (i == hours_of_day[cap - slc:cap])
    week = np.zeros((6, len(test_demand)), dtype=np.uint8)
    for i in range(6):
        week[i] = (i == days_of_week[cap - slc:cap])
    month = np.zeros((11, len(test_demand)), dtype=np.uint8)
    for i in range(1, 12):
        month[i - 1] = (i == months_of_year[cap - slc:cap])

    weekhour = np.zeros((6 * 23, len(test_demand)))
    for i in range(6):
        for j in range(23):
            weekhour[23 * i + j] = week[i] * hour[j]

    trend = np.arange(slc, slc + len(demand))
    temps = []
    num_years = train_through / (24 * 365.24)
    k = int((99 / num_years - 1) / 2 + 0.5)
    print("k =", k)
    for yr in range(int(num_years + 0.5)):
        for shift in range(2*k + 1):
            offset = int((shift + int(yr * 365.24 + 0.5)) * 24 + 0.5)
            temps.append(all_temps[-len(all_demand) - k * 24:][offset + gap: offset + slc + gap])

    preds = []
    for temp in temps:
        temp2 = temp ** 2
        temp3 = temp ** 3
        tempmonth = np.array(list(map(lambda x: temp * x, month)))
        temp2month = np.array(list(map(lambda x: temp2 * x, month)))
        temp3month = np.array(list(map(lambda x: temp3 * x, month)))
        temphour = np.array(list(map(lambda x: temp * x, hour)))
        temp2hour = np.array(list(map(lambda x: temp2 * x, hour)))
        temp3hour = np.array(list(map(lambda x: temp3 * x, hour)))
        test_variables = [trend, *month, *week, *hour, *weekhour, temp, temp2, temp3, *tempmonth, *temp2month, *temp3month,
                          *temphour, *temp2hour, *temp3hour]
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


def get_lossesGEFCom(start=0, zones=None, plot=False, train_through=365*24, delay=0):
    with open("GEFCom2017//GEFCom2017-Qual//GEFCom2017Qual2005-2015.json", "r") as f:
        alldata = json.loads(f.read())

    losses = dict()
    zones = alldata.keys() if zones is None else zones
    for zone_name in zones:
        print(zone_name)
        hours_of_day = np.array(alldata[zone_name]["Hour"], dtype=np.uint8)[start:]
        demand = np.array(alldata[zone_name]["DEMAND"], dtype=np.float64)[start:]
        all_temps = np.array(alldata[zone_name]["DryBulb"], dtype=np.float64)[start % (365 * 24):]
        if start == 0:
            prefix = np.tile(all_temps[:24], 4)
            all_temps = np.concatenate([prefix, all_temps])
        print("start:", alldata[zone_name]["Date"][start])
        months = np.array(list(map(lambda x: int(x[5:7]), alldata[zone_name]["Date"])), dtype=np.uint8)[start:]

        length = len(demand)
        losses[zone_name] = test(demand, hours_of_day, all_temps, months, train_through=train_through, gap=delay, test_through=31*24, plot=plot)

    print(losses)
    return losses


if __name__ == "__main__":
    # losses = get_lossesGEFCom(start=9 * 365 * 24 + 2 * 24 + 31 * 24, plot=True)
    losses = get_lossesGEFCom(start=int(9 * 24 * 365 / 12), delay=52 * 24,  train_through=(10 * 365 + 2) * 24, zones=["ISONE CA"], plot=True)
    print(losses)
