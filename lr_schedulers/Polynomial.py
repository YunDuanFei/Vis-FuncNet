from torch.optim.lr_scheduler import _LRScheduler
from . import config

class PolyAnnealingLR_chen(_LRScheduler):

    def __init__(self, optimizer, T_max, alph, eps, last_epoch=-1):
        self.T_max = T_max
        self.alph = alph
        self.eps = eps
        super(PolyAnnealingLR_chen, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        else:
            return [base_lr*(1-self.last_epoch/self.T_max)**self.alph+self.eps for base_lr in self.base_lrs]


class PolyAnnealingLR_bottou(_LRScheduler):

    def __init__(self, optimizer, nabda, alph, last_epoch=-1):
        self.nabda = nabda
        self.alph = alph
        super(PolyAnnealingLR_bottou, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        else:
            return [base_lr*(1+self.nabda*base_lr*self.last_epoch)**(self.alph) for base_lr in self.base_lrs]


# PolychenLR
def polychen_lr(optimizer):
    scheduler = PolyAnnealingLR_chen(optimizer, T_max=config.polychen_T, alph=config.polychen_alph, eps=1e-08)
    return scheduler

# PolybottouLR
def polybottou_lr(optimizer):
    scheduler = PolyAnnealingLR_bottou(optimizer, nabda=config.polybottou_nabda, alph=config.polybottou_alph)
    return scheduler