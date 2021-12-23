import torch.optim as optim
from . import config

# CosineLR
def cosine_lr(optimizer):
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.cosine_T, eta_min=1e-7)
    return scheduler