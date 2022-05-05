import torch
import torch.nn as nn


def l1_norm(model: nn.Module):
    l1_norm = 0
    for param in model.parameters():
        l1_norm += torch.norm(param, 1)
    return l1_norm


def l2_norm(model: nn.Module):
    l2_norm = 0
    for param in model.parameters():
        l2_norm += torch.norm(param)
    return l2_norm
