from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import math
from . import config

class LWarmupCosineLR(_LRScheduler):
    def __init__(self, optimizer, max_iter, pct_start, warmup_factor, 
                 eta_min=1e-7, last_epoch=-1):
        self.warmup_factor = warmup_factor
        self.warmup_iters = int(pct_start * max_iter)
        self.max_iter, self.eta_min = max_iter, eta_min
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch <= self.warmup_iters:
            alpha = self.last_epoch / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [lr * warmup_factor for lr in self.base_lrs]
        else:
            # print ("after warmup")
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(
                        math.pi * (self.last_epoch - self.warmup_iters) / (self.max_iter - self.warmup_iters))) / 2
                    for base_lr in self.base_lrs]


# LWarmupCosineLR
def lwarmcosine_lr(optimizer):
    scheduler = LWarmupCosineLR(optimizer, max_iter=config.lwarmcosine_max, pct_start=config.lwarmcosine_pct, warmup_factor=config.lwarmcosine_warm)
    return scheduler