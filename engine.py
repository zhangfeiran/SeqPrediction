import math
import sys
import time
import torch
import utils
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def train_one_epoch(encoder, decoder, optimizer0, optimizer1, criterion, data_loader, device, epoch, print_freq,teacher_force=0.5):
    encoder.train()
    decoder.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    losses = []

    for xs, y_hists, y_targs in metric_logger.log_every(data_loader, print_freq, header):
        xs, y_hists, y_targs = xs.to(device), y_hists.to(device), y_targs.to(device)

        input_encoded = encoder(xs)
        y_preds = decoder(input_encoded, y_hists, y_targs,teacher_force)
        loss = criterion(y_preds, y_targs)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        losses.append(loss_value)

        optimizer0.zero_grad()
        optimizer1.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer0.step()
        optimizer1.step()

        metric_logger.update(loss=loss)
        metric_logger.update(lr=optimizer0.param_groups[0]["lr"])
    return losses


@torch.no_grad()
def evaluate(encoder, decoder, dataloader_val, device,return_confusion=False):
    encoder.eval()
    decoder.eval()
    start = time.time()
    confusions = np.zeros((6, 6, 6))  # 6 hours * 6 classes * 6 classes

    for xs, y_hists, y_targs in dataloader_val:
        xs, y_hists, y_targs = xs.to(device), y_hists.to(device), y_targs.to(device)

        input_encoded = encoder(xs)
        y_preds = decoder(input_encoded, y_hists, None, teacher_force=0)

        y_preds, y_targs = y_preds.cpu().numpy(), y_targs.cpu().numpy()

        bins = np.array([0, 35, 75, 115, 150, 250]) / 500
        y_preds, y_targs = np.digitize(y_preds, bins), np.digitize(y_targs, bins)

        for i in range(6):
            confusions[i] += confusion_matrix(y_targs[:, i, :].flatten(), y_preds[:, i, :].flatten(), labels=list(range(1, 7)))

    precisions = np.zeros((6, 6))
    for i in range(6):
        precisions[i] = np.diag(confusions[i]) / (confusions[i].sum(0)+1e-8)
    df = pd.DataFrame(precisions.round(2), index=[f't={i}' for i in range(1, 7)], columns=[f'level={i}' for i in range(1, 7)])

    if return_confusion:
        return confusions,df
    return df