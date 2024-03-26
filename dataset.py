# %%
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils.data


# %%
class AirDataset(torch.utils.data.Dataset):
    def __init__(self, split, T, use_extra=False):
        assert split in {"train", "val", "test"}
        self.split = split
        self.T = T
        df = pd.read_pickle(f"./data/df_{split}.pkl")
        df.reset_index(drop=True, inplace=True)

        df = df.drop(columns="time")
        df = df.clip(upper=500)
        df = df / 500

        self.pm25s = [i for i in range(len(df.columns)) if df.columns[i].startswith("PM2.5")]
        if use_extra:
            self.feats = [i for i in range(len(df.columns)) if not df.columns[i].startswith("PM2.5")]
        else:
            self.feats = [i for i in range(len(df.columns)) if df.columns[i].startswith("PM10") or df.columns[i].startswith("AQI")]
        self.len = len(df) - T - 6 + 1

        self.ts = torch.as_tensor(df.values, dtype=torch.float32)

    def __getitem__(self, idx):
        x = self.ts[idx : idx + self.T, self.feats]
        y_hist = self.ts[idx : idx + self.T, self.pm25s]
        y_targ = self.ts[idx + self.T : idx + self.T + 6, self.pm25s]

        return x, y_hist, y_targ

    def __len__(self):
        return self.len


# %%
class AirDatasetMutation(torch.utils.data.Dataset):
    def __init__(self, split, T, use_extra=False, thresh=0.2):
        assert split in {"val", "test"}
        self.split = split
        self.T = T
        df = pd.read_pickle(f"./data/df_{split}.pkl")
        df.reset_index(drop=True, inplace=True)
        df = df.drop(columns="time")
        df = df.clip(upper=500)
        df = df / 500
        self.pm25s = [i for i in range(len(df.columns)) if df.columns[i].startswith("PM2.5")]
        if use_extra:
            self.feats = [i for i in range(len(df.columns)) if not df.columns[i].startswith("PM2.5")]
        else:
            self.feats = [i for i in range(len(df.columns)) if df.columns[i].startswith("PM10") or df.columns[i].startswith("AQI")]
        mutations = []
        for i in range(len(df) - T - 6 + 1):
            # if df: # i+T-1 dao i+T+6 de min he max cha ju zu gou da
            vals = df[i + T - 1 : i + T + 6].iloc[:, self.pm25s].values
            if vals.max() - vals.min() > thresh:
                mutations.append(i)

        self.len = len(mutations)
        self.mutations = mutations
        self.ts = torch.as_tensor(df.values, dtype=torch.float32)

    def __getitem__(self, idx):
        idx = self.mutations[idx]
        x = self.ts[idx : idx + self.T, self.feats]
        y_hist = self.ts[idx : idx + self.T, self.pm25s]
        y_targ = self.ts[idx + self.T : idx + self.T + 6, self.pm25s]

        return x, y_hist, y_targ

    def __len__(self):
        return self.len


def collate_fn(batch):
    xs, y_hists, y_targs = tuple(zip(*batch))
    xs = torch.stack(xs, dim=0)
    y_hists = torch.stack(y_hists, dim=0)
    y_targs = torch.stack(y_targs, dim=0)
    return xs, y_hists, y_targs

