import torch.optim as optim
from . import config

# TriangularLR
def triangular_lr(optimizer):
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=config.triangular_base_lr, max_lr=config.triangular_max_lr, 
    step_size_up=config.triangular_size, mode=config.triangular_mode)
    return scheduler

# Triangular2LR
def triangular2_lr(optimizer):
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=config.triangular_base_lr, max_lr=config.triangular_max_lr, 
    step_size_up=config.triangular_size, mode=config.triangular2_mode)
    return scheduler

# TriangularexpLR
def triangularexp_lr(optimizer):
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=config.triangular_base_lr, max_lr=config.triangular_max_lr, 
    step_size_up=config.triangular_size, mode=config.triangularexp_mode, gamma=config.triangularexp_gamma)
    return scheduler