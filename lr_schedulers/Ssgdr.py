from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from . import config


class SSGDRAnnealingLR(_LRScheduler):

    def __init__(self, optimizer, T_max, stepsize, eps=1e-6, mode='without_decay', nabda=None, last_epoch=-1):
        self.stepsize = stepsize
        self.T_max = T_max
        self.eps = eps
        self.mode = mode
        self.nabda = nabda
        super(SSGDRAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        clc_num, iter_num = divmod(self.last_epoch, self.stepsize)
        if self.mode == 'without_decay':
            if iter_num < 0.5*self.stepsize:
                return [base_lr for base_lr in self.base_lrs]
            else:
                return [base_lr*np.cos(np.pi*(iter_num-0.5*self.stepsize)/(self.stepsize)) for base_lr in self.base_lrs]
        if self.mode == 'with_decay':
            alph = self.nabda**clc_num
            if iter_num < 0.5*self.stepsize:
                return [base_lr*alph for base_lr in self.base_lrs]
            else:
                return [alph*base_lr*np.cos(np.pi*(iter_num-0.5*self.stepsize)/(self.stepsize)) for base_lr in self.base_lrs]

# SSGDRLR 
def ssgdr_lr(optimizer):
    scheduler = SSGDRAnnealingLR(optimizer, T_max=config.ssgdr_T, stepsize=config.ssgdr_size, mode='without_decay')
    return scheduler

# SSGDRLR with decay
def ssgdrd_lr(optimizer):
    scheduler = SSGDRAnnealingLR(optimizer, T_max=config.ssgdr_T, stepsize=config.ssgdr_size, mode='with_decay', nabda=config.ssgdr_nabda)
    return scheduler