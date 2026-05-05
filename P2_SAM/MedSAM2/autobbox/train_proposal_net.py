"""
train_proposal_net.py
=====================
Train proposal network (recall-focused)
"""

import torch
import torch.nn.functional as F
import numpy as np
import scipy.ndimage as ndi
from model import Small3DUNet


def make_soft_target(gt):
    tumor = (gt > 0).astype(np.float32)
    dist = ndi.distance_transform_edt(1 - tumor)
    sigma = 3
    return np.exp(-dist**2 / (2 * sigma**2))


def recall_loss(pred, target):
    eps = 1e-6
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    return 1 - tp / (tp + fn + eps)


def combined_loss(pred, target):
    bce = F.binary_cross_entropy(pred, target)
    rec = recall_loss(pred, target)
    return 0.3 * bce + 0.7 * rec


def train_step(model, optimizer, x, gt):
    model.train()

    target = make_soft_target(gt)
    target = torch.tensor(target).unsqueeze(0).unsqueeze(0).float()

    pred = model(x)

    loss = combined_loss(pred, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
