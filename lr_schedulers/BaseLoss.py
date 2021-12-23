import torch.optim as optim
from . import config


# BaseLossLR
def baseloss_lr(optimizer):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.baseloss_factor, patience=config.baseloss_patience,verbose=False, 
    threshold=0.0001, threshold_mode='rel', cooldown=config.baseloss_cooldown, min_lr=config.baseloss_min, eps=1e-08)
    return scheduler