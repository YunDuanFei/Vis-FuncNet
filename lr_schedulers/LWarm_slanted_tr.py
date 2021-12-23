from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from . import config

class LWarmupSlantedTr(_LRScheduler):
    def __init__(self, optimizer, T_max, cut_frac, ratio, last_epoch=-1):
        self.cut = np.floor(T_max*cut_frac)
        self.ratio = ratio
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.cut:
            p = self.last_epoch/self.cut
        else:
            p = 1-(self.last_epoch-self.cut)/(self.cut*(self.ratio-1))
        return [base_lr*(1+p*(self.ratio-1))/self.ratio for base_lr in self.base_lrs]

# LWarmupSlantedTriangularLR
def lwarmslanted_tr_lr(optimizer):
    scheduler = LWarmupSlantedTr(optimizer, T_max=config.lwarmslanted_tr_T, cut_frac=config.lwarmslanted_tr_cut, ratio=config.lwarmslanted_tr_ratio)
    return scheduler