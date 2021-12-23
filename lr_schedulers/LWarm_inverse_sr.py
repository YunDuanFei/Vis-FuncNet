from torch.optim.lr_scheduler import _LRScheduler
from . import config


class LWarmupInversesrLR(_LRScheduler):
    def __init__(self, optimizer, max_iter, pct_start, d_model, muiti_fac, last_epoch=-1):
        self.d_model = d_model
        self.warmup_iters = int(pct_start * max_iter)
        self.muiti_fac = muiti_fac
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        lr_mul = self.muiti_fac*(self.d_model ** -0.5) * min(self.last_epoch ** (-0.5), self.last_epoch * self.warmup_iters ** (-1.5))
        return [lr_mul*base_lr for base_lr in self.base_lrs]


# LinearWarmInversesr_lr
def lwarminversesr_lr(optimizer):
    scheduler = LWarmupInversesrLR(optimizer, max_iter=config.lwarminversesr_T, pct_start=config.lwarminversesr_pct, 
        d_model=config.lwarminversesr_d, muiti_fac=config.lwarminversesr_muiti)
    return scheduler