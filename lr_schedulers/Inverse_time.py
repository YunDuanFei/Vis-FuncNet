from torch.optim.lr_scheduler import _LRScheduler
from . import config


class InversetimeAnnealingLR(_LRScheduler):

    def __init__(self, optimizer, nabda, last_epoch=-1):
        self.nabda = nabda
        super(InversetimeAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        else:
            return [base_lr*(1+self.nabda*base_lr*self.last_epoch)**(-1) for base_lr in self.base_lrs]


# InversetimeLR
def inversetime_lr(optimizer):
    scheduler = InversetimeAnnealingLR(optimizer, nabda=config.inversetime_nabda)
    return scheduler