from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from . import config

class LWarmup1Cycle(_LRScheduler):
    def __init__(self, optimizer, T_max, prcnt, div, last_epoch=-1):
        self.step_len = np.floor(0.5*T_max*(1-prcnt/100))
        self.div = div
        self.T_max = T_max
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch == 0:
            return [base_lr/self.div for base_lr in self.base_lrs]
        elif self.last_epoch >= 2*self.step_len:
            ratio = (self.last_epoch - 2 * self.step_len) / (self.T_max - 2 * self.step_len)
            return [(base_lr / self.div) * (1- ratio * (1 - 1/self.div)) for base_lr in self.base_lrs]
        elif self.last_epoch > self.step_len:
            ratio = 1- (self.last_epoch -self.step_len)/self.step_len
            return [base_lr * (1 + ratio * (self.div - 1)) / self.div for base_lr in self.base_lrs]
        else:
            ratio = self.last_epoch/self.step_len
            return [base_lr * (1 + ratio * (self.div - 1)) / self.div for base_lr in self.base_lrs]

# LWarmup1cycleLR
def lwarm1cycle_lr(optimizer):
    scheduler = LWarmup1Cycle(optimizer, T_max=config.lwarm1cycle_T, prcnt=config.lwarm1cycle_prcnt, div=config.lwarm1cycle_div)
    return scheduler