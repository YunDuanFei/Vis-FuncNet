import torch.optim as optim
from . import config

# Constant_lr
def constant_lr(optimizer):
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.constant_size, gamma=config.constant_gamma)
    return scheduler