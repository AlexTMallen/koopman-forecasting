#!/usr/bin/env python
# coding: utf-8

import json
import numpy as np
import pandas as pd
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


def test(all_df, train_start, train_through, test_length, gap=0, temp_years=None, plot=False):
    variables = pd.DataFrame(all_df.temp)
    variables["temp2"] = variables.temp ** 2
    variables["temp3"] = variables.temp ** 3
    variables["trend"] = variables.index

    for i in range(2, 13):
        variables[f"m{i}"] = (i == all_df.month) * 1  # converts to int
        variables[f"tempmonth{i}"] = variables.temp * variables[f"m{i}"]
        variables[f"temp2month{i}"] = variables.temp2 * variables[f"m{i}"]
        variables[f"temp3month{i}"] = variables.temp3 * variables[f"m{i}"]

    for i in range(2, 25):
        variables[f"h{i}"] = (i == all_df.hour_end) * 1
        variables[f"temphour{i}"] = variables.temp * variables[f"h{i}"]
        variables[f"temp2hour{i}"] = variables.temp2 * variables[f"h{i}"]
        variables[f"temp3hour{i}"] = variables.temp3 * variables[f"h{i}"]

    for i in range(1, 7):
        variables[f"w{i}"] = (i == all_df.day_of_week) * 1

    for i in range(1, 7):
        for j in range(2, 25):
            variables[f"wh{i}_{j}"] = variables[f"w{i}"] * variables[f"h{j}"]

    cap = train_through
    train_df = all_df.iloc[train_start:cap]
    train_vars = variables.iloc[train_start:cap]

    reg = linear_model.LinearRegression()

    reg.fit(train_vars, train_df.load)

    if plot:
        fit = reg.predict(variables)
        plt.plot(all_df.load.values)
        plt.plot(fit)
        plt.axvline(train_through, color="k")
        plt.show()

    #  ###---TEST---###
    start = train_through + gap
    cap = train_through + test_length + gap
    test_demand = all_df.load.iloc[start:cap]

    temps = []
    years = np.unique(train_df.year) if temp_years is None else temp_years
    num_years = len(years)
    k = int((99 / num_years - 1) / 2 + 0.5)
    print("k =", k)
    for yr in years:
        this_year = all_df.iloc[:train_through][all_df.year.iloc[:train_through] == yr]
        for shift in range(-k, k + 1):
            delta = pd.Timedelta(days=shift)
            dates = all_df.date.iloc[start:cap] + delta
            test_days_of_year = np.unique(list(date.timetuple().tm_yday for date in dates))
            include = list(day in test_days_of_year for day in this_year.day_of_year)
            temps.append(this_year.temp[include].values)

    if plot:
        plt.plot(np.array(temps).T, linewidth=0.5)
        plt.plot(variables.temp[start:cap].values, linewidth=2, color="orange")
        plt.show()

    preds = []
    for i, temp in enumerate(temps):
        test_vars = variables.iloc[start:cap].copy()

        test_vars.loc[:, "temp"] = temp
        test_vars.loc[:, "temp2"] = temp ** 2
        test_vars.loc[:, "temp3"] = temp ** 3

        for i in range(2, 13):
            test_vars.loc[:, f"tempmonth{i}"] = test_vars.temp * test_vars[f"m{i}"]
            test_vars.loc[:, f"temp2month{i}"] = test_vars.temp2 * test_vars[f"m{i}"]
            test_vars.loc[:, f"temp3month{i}"] = test_vars.temp3 * test_vars[f"m{i}"]

        for i in range(2, 25):
            test_vars.loc[:, f"temphour{i}"] = test_vars.temp * test_vars[f"h{i}"]
            test_vars.loc[:, f"temp2hour{i}"] = test_vars.temp2 * test_vars[f"h{i}"]
            test_vars.loc[:, f"temp3hour{i}"] = test_vars.temp3 * test_vars[f"h{i}"]

        preds.append(reg.predict(test_vars))

    preds = np.array(preds)

    if plot:
        pred = np.mean(preds, axis=0)
        std = np.std(preds, axis=0)
        plt.plot(test_demand.values)
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

    return pinball_loss(test_demand.values, quant_preds, quantiles)


def get_lossesGEFCom(train_through_years, test_length, zones=None, start_date=None, delay_days=0, temp_years=None, plot=False):
    with open("GEFCom2017//GEFCom2017-Qual//GEFCom2017Qual2005-2015.json", "r") as f:
        alldata = json.loads(f.read())

    losses = dict()
    zones = alldata.keys() if zones is None else zones
    for zone_name in zones:
        print(zone_name)
        zone_data = alldata[zone_name]
        all_df = pd.DataFrame({"date": zone_data["Date"], "load": zone_data["DEMAND"], "temp": zone_data["DryBulb"]})
        all_df.date = pd.to_datetime(all_df.date)
        all_df["day_of_week"] = list((i // 24) % 7 for i in range(len(all_df)))
        all_df["hour_end"] = list(i % 24 + 1 for i in range(len(all_df)))
        all_df["month"] = list(all_df.date.loc[i].month for i in all_df.index)
        all_df["day_of_year"] = list(date.timetuple().tm_yday for date in all_df.date)
        all_df["year"] = list(date.timetuple().tm_year for date in all_df.date)

        bad_idxs = all_df.load.isnull().to_numpy().nonzero()[0]
        all_df.loc[bad_idxs, "load"] = (all_df.load.values[bad_idxs - 1] + all_df.load.values[bad_idxs + 1]) / 2

        start_date = all_df.date.iloc[0] if start_date is None else start_date
        train_start = all_df.index[start_date == all_df.date][0]
        train_through = all_df[all_df.date == start_date.replace(year=start_date.year + train_through_years)].index[0]
        losses[zone_name] = test(all_df, train_start, train_through, test_length, gap=delay_days * 24, temp_years=temp_years, plot=plot)
    print(losses)
    return losses


if __name__ == "__main__":
    # losses = get_lossesGEFCom(start=9 * 365 * 24 + 2 * 24 + 31 * 24, plot=True)
    losses = get_lossesGEFCom(10, 31 * 24, start_date=pd.Timestamp("2005-10-01"), zones=["ISONE CA"],
                              delay_days=0, temp_years=[2014], plot=True)
    print(losses)
