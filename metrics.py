import torch
from scipy.stats import spearmanr


def calculate_mae(y_true, y_pred):
    mae = torch.mean(torch.abs(y_true - y_pred)).item()
    return mae


def calculate_spearman(y_true, y_pred):
    y_true = (torch.exp(y_true) - 1).cpu().numpy()
    y_pred = (torch.exp(y_pred.detach()) - 1).cpu().numpy()
    spearman_corr, _ = spearmanr(y_true, y_pred)
    return spearman_corr


def calculate_r_squared(y_true, y_pred):
    mean_y_true = torch.mean(y_true)
    ss_total = torch.sum((y_true - mean_y_true) ** 2)
    ss_residual = torch.sum((y_true - y_pred) ** 2)
    r_squared = 1 - ss_residual / ss_total
    r_squared = r_squared.item()
    return r_squared
