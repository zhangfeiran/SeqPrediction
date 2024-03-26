import os
import sys
import time
from importlib import reload

import numpy as np
import pandas as pd
import torch
import torch.utils.data

import model
import dataset
import engine
import utils
import preprocess

try:
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    torch.cuda.get_device_properties(device)
except:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparams
torch.backends.cudnn.benchmark = True
T = 6
lr = 0.002
weight_decay = 0
batch_size = 64

# model
encoder = model.Encoder(input_size=22, T=T)
decoder = model.Decoder(T=T)
encoder.to(device)
decoder.to(device)

# preprocessing
if not os.path.exists("./data/df_test.pkl"):
    preprocess.preprocess()


checkpoint_path = "./data/checkpoint_zfr.pt"
if not os.path.exists(checkpoint_path):
    print("no checpoint_zfr.pt, start training...")

    dataset_train = dataset.AirDataset("train", T, use_extra=False)

    try:
        l = len(dataset_train)
    except:
        print("no data for training, exiting")
        exit()

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn, num_workers=0, pin_memory=True,
    )

    optimizer0 = torch.optim.Adam([p for p in encoder.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    optimizer1 = torch.optim.Adam([p for p in decoder.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)

    lr_scheduler0 = torch.optim.lr_scheduler.StepLR(optimizer0, step_size=3, gamma=0.1)
    lr_scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=3, gamma=0.1)
    criterion = torch.nn.MSELoss()

    num_epoch = 7
    for epoch in range(num_epoch):
        engine.train_one_epoch(
            encoder, decoder, optimizer0, optimizer1, criterion, dataloader_train, device, epoch, print_freq=100, teacher_force=1
        )
        lr_scheduler0.step()
        lr_scheduler1.step()

    checkpoint_path = "./data/checkpoint_zfr.pt"
    torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict()}, checkpoint_path)

else:
    print("loading checkpoint_zfr.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

print("start testing")
dataset_test = dataset.AirDataset("test", T, use_extra=False)
dataloader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn, num_workers=0, pin_memory=True,
)
try:
    l = len(dataset_test)
except:
    print("no data for testing, exiting")
    exit()
a = engine.evaluate(encoder, decoder, dataloader_test, device,)
print(a)
