from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from . import config


class CosinePowerAnnealingLR(_LRScheduler):

    def __init__(self, optimizer, p, T_max, lr_min=1e-7, last_epoch=-1):
        self.p = p
        self.lr_min = lr_min
        self.T_max = T_max
        super(CosinePowerAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        else:
            lrp = (self.p**(0.5*(1+np.cos(np.pi*self.last_epoch/self.T_max))+1)-self.p)/(self.p**2-self.p)
            return [self.lr_min+(base_lr-self.lr_min)*lrp for base_lr in self.base_lrs]


# CosinepowerLR 
def cosinepower_lr(optimizer):
    scheduler = CosinePowerAnnealingLR(optimizer, p=config.cosinepower_p, T_max=config.cosinepower_T)
    return scheduler