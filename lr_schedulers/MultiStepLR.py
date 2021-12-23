import torch.optim as optim
from . import config

# MultiStepLR
def multistep_lr(optimizer):
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.multistep_milestones, gamma=config.multistep_gamma)
    return scheduler