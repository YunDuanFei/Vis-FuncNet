import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from . import config


class SGDRDecayAnnealingLR(_LRScheduler):

    def __init__(self, optimizer, T_max, num_iter, alph, last_epoch=-1):
        self.T_max = T_max
        self.num_iter=num_iter
        self.alph=alph
        super(SGDRDecayAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs 
        else:
            cos_inner = np.pi*(self.last_epoch%(self.T_max//self.num_iter))
            cos_inner /= self.T_max//self.num_iter
            cos_out = np.cos(cos_inner)+1
            return [base_lr/2*cos_out*self.alph**(self.last_epoch//(self.T_max//self.num_iter)) for base_lr in self.base_lrs]


# SGDRLR Equally spaced
def sgdres_lr(optimizer):
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.sgdres_T, T_mult=config.sgdres_Tmul)
    return scheduler

# SGDRLR not Equally spaced
def sgdrnes_lr(optimizer):
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.sgdrnes_T, T_mult=config.sgdrnes_Tmul)
    return scheduler

# SGDRLR with decay
def sgdrd_lr(optimizer):
    scheduler = SGDRDecayAnnealingLR(optimizer, T_max=config.sgdrd_T, num_iter=config.sgdrd_num, alph=config.sgdrd_alph)
    return scheduler