import collections
import datetime
import gzip
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn


def preprocess():
    print('start preprocessing')
    fmt = "%Y%m%d"
    types = ["AQI", "CO", "NO2", "O3", "PM10", "PM2.5", "SO2"]
    locs = [
        "date",
        "hour",
        "type",
        "东四",
        "天坛",
        "官园",
        "万寿西宫",
        "奥体中心",
        "农展馆",
        "万柳",
        "北部新区",
        "丰台花园",
        "云岗",
        "古城",
    ]

    def f2p(f):
        ff = f.split("_")
        return f"./data/{ff[2][:4]}/{ff[0]}_{ff[1]}_{ff[2][:8]}.csv"

    def d2p(d):
        if type(d) == pd.Timestamp:
            d = d.strftime(fmt)
        return f"./data/{d[:4]}/beijing_all_{d}.csv", f"./data/{d[:4]}/beijing_extra_{d}.csv"

    def ds2p(ds):
        return [p for d in ds for p in d2p(d)]

    # # rename folders
    for folder in os.listdir("./data"):
        if folder.startswith("beijing") and os.path.isdir(f"./data/{folder}"):
            os.rename(f"./data/{folder}", f"./data/{folder[8:12]}")
            print(f"rename from ./data/{folder} to ./data/{folder[8:12]}")

    # # gzip wtf
    try:
        a = gzip.open(d2p("20141231")[0])
        b = a.read()
        a.close()
        with open(d2p("20141231")[0], "w", encoding="utf8") as f:
            f.write(b.decode("utf8"))
    except Exception as e:
        print(e)

    # # delete blank
    missed = [
        "./data/2014/.DS_Store",
        "./data/2015/.DS_Store",
    ]
    missed += ds2p(["20161230", "20161231"])  # blank
    missed += ds2p(pd.date_range("20170519", "20170530"))  # http error
    missed += ds2p(pd.date_range("20170702", "20170708"))

    for file in missed:
        try:
            os.remove(file)
            print(f"blank file {file} removed")
        except Exception as e:
            # print(e)
            # print('already removed')
            pass

    # # Read All
    print("reading csv...")
    dfs = []
    for year in range(2014, 2021):
        base = f"./data/{year}/"
        if not os.path.isdir(base):
            continue
        for file in os.listdir(base):
            df = pd.read_csv(base + file)
            df = df[[not i.endswith("h") for i in df.type]]
            dfs.append(df)

    # # choose subset locs
    print("subsetting...")
    dfs = [df[locs] for df in dfs]
    # # delete missings
    dfs = [df for df in dfs if df.shape[0] in (72, 96)]
    dfs = [df for df in dfs if df.isna().values.sum() / df.shape[0] / df.shape[1] < 0.2]
    dates = collections.Counter([df.date[0] for df in dfs])
    dfs = [df for df in dfs if dates[df.date[0]] == 2]

    # # concat and split
    try:
        df_all = pd.concat(dfs)
    except:
        print('no data to preprocess, exiting')
        exit()
    df_all = df_all.sort_values(by=["date", "hour", "type"])
    df_all = df_all.pivot_table(index=["date", "hour"], columns=["type"], values=locs)
    df_all.columns = [i[1] + "_" + i[0] for i in df_all.columns.values]
    df_all["time"] = [i[0] * 100 + i[1] for i in df_all.index.values]
    df_all.reset_index(drop=True, inplace=True)
    df_all = df_all.interpolate()

    df_test = df_all[df_all.time < 2014123200]
    df_train = df_all[(2015000000 < df_all.time) & (df_all.time < 2019123200)]
    df_val = df_all[2020000000 < df_all.time]

    pd.to_pickle(df_test, "./data/df_test.pkl")
    pd.to_pickle(df_val, "./data/df_val.pkl")
    pd.to_pickle(df_train, "./data/df_train.pkl")
    print("pickle saved")
