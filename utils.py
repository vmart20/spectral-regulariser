import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False

def get_split(y, nclass, seed=0, train_prc=0.6, val_prc=0.35, percls_trn=10):
    y = y.cpu()
    g = torch.Generator().manual_seed(seed)
    if percls_trn is None or percls_trn == "None":
        percls_trn = int(round(train_prc * len(y) / nclass))
    indices = []
    for i in range(nclass):
        index = (y == i).nonzero(as_tuple=True)[0]
        perm = torch.randperm(index.size(0), generator=g)
        indices.append(index[perm])
    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)
    val_index = torch.cat([i[percls_trn:percls_trn+int(round(val_prc * (i.size(0)-percls_trn)))] for i in indices], dim=0)
    test_index = torch.cat([i[percls_trn+int(round(val_prc * (i.size(0)-percls_trn))):] for i in indices], dim=0)
    return train_index, val_index, test_index

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def train_model(net, optimizer, evaluation, epoch, train, valid, test, y, net_args, early_stop=True, loss_type="cross_entropy", gamma =0):
    res = []
    epoch_times = []
    best_state_dict = {}
    counter = 0
    best_val_acc = -1

    for idx in range(epoch):
        net.train()
        optimizer.zero_grad()
        logits = net(*net_args)
        
        loss = F.cross_entropy(logits[train], y[train])
        
        if gamma != 0:
            loss = loss + gamma * net.regularisation()

        loss.backward()
        optimizer.step()

        net.eval()
        with torch.no_grad():
            logits = net(*net_args)

            train_loss = F.cross_entropy(logits[train], y[train]).item()
            test_loss = F.cross_entropy(logits[test], y[test]).item()
            val_loss = F.cross_entropy(logits[valid], y[valid]).item()
            train_acc = evaluation(logits[train].cpu(), y[train].cpu()).item()
            val_acc = evaluation(logits[valid].cpu(), y[valid].cpu()).item()
            test_acc = evaluation(logits[test].cpu(), y[test].cpu()).item()

        print(idx, float(loss), train_acc, val_loss, val_acc, test_acc)
        
        if val_acc > best_val_acc:
            counter = 0
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_state_dict = net.state_dict()
            res = {"loss": float(loss), "train_acc": train_acc, "val_loss": val_loss, "test_loss": test_loss, "test_acc": test_acc, "train_loss": train_loss, "val_acc": val_acc, "gap": test_loss - train_loss}
        else:
            counter += 1


        if counter == 200 and early_stop:
            break


    return res, best_val_acc, best_test_acc, best_state_dict
