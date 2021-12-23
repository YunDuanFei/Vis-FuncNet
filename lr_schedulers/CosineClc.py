import torch.optim as optim
from . import config

# CosineClcLR
def cosineclc_lr(optimizer):
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.cosineclc_T, eta_min=1e-7, last_epoch=-1)
    return scheduler
