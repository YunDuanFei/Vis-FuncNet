import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from . import config


class TriangleAnnealingLR(_LRScheduler):

    def __init__(self, optimizer, multi_steps, gamma, cycle_len, last_epoch=-1):
        self.multi_steps = multi_steps
        self.step_counter = 0
        self.stepping = True
        self.min_lr = 1e-7
        self.decayfactor = gamma
        self.count_cycles = 0
        self.warm_up_interval = 1
        self.counter = 0
        self.cycle_len = cycle_len
        super(TriangleAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            self.min_lr = self.base_lrs[0]
            return self.base_lrs 
        else:
            if self.last_epoch%self.multi_steps[self.step_counter]==0 and self.last_epoch>1 and self.stepping:
                self.min_lr = self.min_lr/self.decayfactor
                self.count_cycles = 0
                if self.step_counter < len(self.multi_steps)-1:
                    self.step_counter += 1
                else:
                    self.stepping = False
            current_lr = self.min_lr
            if self.count_cycles < self.warm_up_interval:
                self.count_cycles += 1
                if self.count_cycles == self.warm_up_interval:
                    self.warm_up_interval = 0
            else:
                if self.counter >= self.cycle_len:
                    self.counter = 0
                current_lr = round(self.min_lr*self.cycle_len-self.counter*self.min_lr, 5)
                self.counter += 1
                self.count_cycles += 1
            return [current_lr]



# TriangleLR with decay
def triangle_lr(optimizer):
    scheduler = TriangleAnnealingLR(optimizer, multi_steps=config.triangle_multi, gamma=config.triangle_gamma, cycle_len=config.triangle_clen)
    return scheduler