from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from . import config

class LWarmupLongTrap(_LRScheduler):
    def __init__(self, optimizer, T_max, t_up, t_down, last_epoch=-1):
        self.tup = np.floor(T_max*t_up)
        self.tdown = np.floor(T_max*t_down)
        self.T_max = T_max
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch <= self.tup:
            p = self.last_epoch/self.tup
        elif self.tup < self.last_epoch <= self.tdown:
            p = 1
        else:
            p = (self.last_epoch-self.T_max)/(self.tdown-self.T_max)
        return [base_lr*p for base_lr in self.base_lrs]

# LWarmupLongTrapezoidLR
def lwarmlong_trap_lr(optimizer):
    scheduler = LWarmupLongTrap(optimizer, T_max=config.lwarmlong_trap_T, t_up=config.lwarmlong_trap_up, t_down=config.lwarmlong_trap_down)
    return scheduler