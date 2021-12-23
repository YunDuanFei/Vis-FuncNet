from torch.optim.lr_scheduler import _LRScheduler
from . import config

class LinearAnnealingLR(_LRScheduler):

    def __init__(self, optimizer, iter_mid, lr_mid, last_epoch=-1):
        self.lr_mid = lr_mid
        self.iter_mid=iter_mid
        super(LinearAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        if self.last_epoch < self.iter_mid:
            return [self.last_epoch*(self.lr_mid-base_lr)/self.iter_mid+base_lr for base_lr in self.base_lrs]  
        else:
            return [self.lr_mid]



# LinearLR
def linear_lr(optimizer):
    scheduler = LinearAnnealingLR(optimizer, iter_mid=int(config.linear_size*0.8), lr_mid=config.linear_low_lr)
    return scheduler