from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from . import config

class LWarmupPolyLR(_LRScheduler):
    def __init__(self, optimizer, T_max, pct_start, warmup_factor, power, eta_min=1e-7):
        self.warmup_factor = warmup_factor
        self.warmup_iters = int(pct_start * T_max)
        self.power = power
        self.T_max, self.eta_min = T_max, eta_min
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch <= self.warmup_iters:
            alpha = self.last_epoch / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [lr * warmup_factor for lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) *
                    math.pow(1 - (self.last_epoch - self.warmup_iters) / (self.T_max - self.warmup_iters),
                             self.power) for base_lr in self.base_lrs]


# LWarmupPolyLR
def lwarmpoly_lr(optimizer):
    scheduler = LWarmupPolyLR(optimizer, max_iter=config.lwarmpoly_max, pct_start=config.lwarmpoly_pct, warmup_factor=config.lwarmpoly_warm, power=config.lwarmpoly_pow)
    return scheduler