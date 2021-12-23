from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from . import config


class LinearcosineAnnealingLR(_LRScheduler):

    def __init__(self, optimizer, T_max, last_epoch=-1):
        self.T_max = T_max
        super(LinearcosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        else:
            ld = 1-self.last_epoch/self.T_max
            cd = 1+np.cos(np.pi*self.last_epoch/self.T_max)
            return [base_lr*ld*cd for base_lr in self.base_lrs]

class LinearcosineNoiseAnnealingLR(_LRScheduler):

    def __init__(self, optimizer, T_max, n_d, last_epoch=-1):
        self.T_max = T_max
        self.n_d = n_d
        super(LinearcosineNoiseAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        else:
            rand_data = np.random.normal(0, 1/(1+self.last_epoch)**0.55, 1)[0]/self.n_d
            ld = 1-self.last_epoch/self.T_max
            cd = 1+np.cos(np.pi*self.last_epoch/self.T_max)
            return [base_lr*(ld+rand_data)*cd+0.0001 for base_lr in self.base_lrs]

# LinearcosineLR
def linearcosine_lr(optimizer):
    scheduler = LinearcosineAnnealingLR(optimizer, T_max=config.linearcosine_T)
    return scheduler

# LinearcosineLR with noise
def linearcosinenoise_lr(optimizer):
    scheduler = LinearcosineNoiseAnnealingLR(optimizer, T_max=config.linearcosine_T, n_d=config.linearcosine_noise_decay)
    return scheduler