import torch.optim as optim
from . import config

# StepLR
def step_lr(optimizer):
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.step_gamma)
    return scheduler