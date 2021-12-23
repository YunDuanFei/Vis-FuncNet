from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from . import config


class SineAnnealingLR(_LRScheduler):

    def __init__(self, optimizer, stepsize, eps=1e-6, mode='SIN', nabda=None, last_epoch=-1):
        self.stepsize = stepsize
        self.eps = eps
        self.mode = mode
        self.nabda = nabda
        super(SineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        else:
            clr = np.abs(np.sin(np.pi*self.last_epoch/(2*self.stepsize)))
            if self.mode == 'SIN':
                return [clr*base_lr+self.eps for base_lr in self.base_lrs]
            elif self.mode == 'SIN2':
                alph = 1/2**(np.floor(self.last_epoch/(2*self.stepsize)))
                return [alph*clr*base_lr+self.eps for base_lr in self.base_lrs]
            elif self.mode == 'SINEXP':
                alph = self.nabda**self.last_epoch
                return [alph*clr*base_lr+self.eps for base_lr in self.base_lrs]
            else:
                raise NotImplementedError('Not Implementaion Yet')

# SineLR 
def sine_lr(optimizer):
    scheduler = SineAnnealingLR(optimizer, stepsize=config.sine_size, mode='SIN')
    return scheduler

# Sine2LR 
def sine2_lr(optimizer):
    scheduler = SineAnnealingLR(optimizer, stepsize=config.sine_size, mode='SIN2')
    return scheduler

# SineExpLR 
def sineexp_lr(optimizer):
    scheduler = SineAnnealingLR(optimizer, stepsize=config.sine_size, mode='SINEXP', nabda=config.sine_nabda)
    return scheduler